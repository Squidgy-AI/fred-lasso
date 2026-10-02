"""Client for the team's VSS backend - the pre-built search/ingest API.

Mirrors the endpoints documented in the challenge's Cursor skills (retrieval/login,
retrieval/search, retrieval/agent-qa, ingest/upload-video) so our app and the skills
agree about the stack. Credentials come from the team config and are never logged.
"""
from __future__ import annotations

import os
import time

import requests

from . import config

_token_cache = {"token": None, "expires": 0.0}


def backend() -> str:
    return config.require("INGRESS_URL").rstrip("/")


def login(force: bool = False) -> str:
    """JWT from POST /api/v1/auth/login, cached for the session."""
    now = time.time()
    if not force and _token_cache["token"] and now < _token_cache["expires"]:
        return _token_cache["token"]
    resp = requests.post(
        f"{backend()}/api/v1/auth/login",
        json={"username": config.require("USERNAME"),
              "password": config.require("PASSWORD")},
        timeout=60,
    )
    resp.raise_for_status()
    token = resp.json()["access_token"]
    _token_cache.update({"token": token, "expires": now + 1800})
    return token


def _headers() -> dict:
    return {"Authorization": f"Bearer {login()}"}


def _request(method: str, path: str, **kwargs):
    url = f"{backend()}{path}"
    resp = requests.request(method, url, headers=_headers(), timeout=kwargs.pop("timeout", 120), **kwargs)
    if resp.status_code == 401:  # token expired mid-session
        login(force=True)
        resp = requests.request(method, url, headers=_headers(), timeout=120, **kwargs)
    resp.raise_for_status()
    return resp.json()


def search(query: str, top_k: int = 15, llm_top_n: int = 3, min_similarity: float = 0.3,
           tags=None, time_filter: str = "all", metadata_filters=None,
           include_public: bool = True, hybrid_text_weight: float = None) -> dict:
    """Hybrid caption+visual search over the team's indexed segments."""
    payload = {
        "query": query, "top_k": top_k, "llm_top_n": llm_top_n,
        "min_similarity": min_similarity, "tags": tags or [],
        "time_filter": time_filter, "metadata_filters": metadata_filters or {},
        "include_public": include_public,
    }
    if hybrid_text_weight is not None:
        payload["hybrid_text_weight"] = hybrid_text_weight
    return _request("POST", "/api/v1/search", json=payload)


def agent_ask(question: str, top_k: int = 10, original_video: str = None) -> dict:
    """Grounded answer with evidence, rather than a list of hits."""
    payload = {"question": question, "top_k": top_k}
    if original_video:
        payload["original_video"] = original_video
    return _request("POST", "/api/v1/agent/ask", json=payload)


def dashboard_stats(scope: str = "mine") -> dict:
    return _request("GET", f"/api/v1/dashboard/stats?scope={scope}")


def explore(scope: str = "all", limit: int = 100, offset: int = 0) -> dict:
    return _request("GET", f"/api/v1/videos/explore?scope={scope}&limit={limit}&offset={offset}")


def ingest_config() -> dict:
    return _request("GET", "/api/v1/metadata/ingest-config")


def app_config() -> dict:
    return _request("GET", "/api/v1/config")


def max_upload_mb(default: int = 25) -> int:
    try:
        cfg = app_config()
        value = (cfg.get("app") or {}).get("max_upload_size_mb")
        return int(value) if value else default
    except Exception:  # noqa: BLE001
        return default


def upload_video(path: str, custom_prompt: str = None, scenario: str = None,
                 camera_id: str = None, capture_type: str = None,
                 location: str = None, tags: str = None,
                 is_public: bool = True) -> dict:
    """POST /api/v1/videos/upload. One file per request, subject to the size cap.

    The organisers' guidance is that re-ingest is the normal path; use this only for
    footage you own and have cleared, and check max_upload_mb() first.
    """
    size_mb = os.path.getsize(path) / 1e6
    cap = max_upload_mb()
    if size_mb > cap:
        raise ValueError(
            f"{os.path.basename(path)} is {size_mb:.1f}MB but the backend cap is {cap}MB. "
            f"Split it first with fred.clips.split_for_upload()."
        )
    form = {"is_public": str(bool(is_public)).lower()}
    for key, value in (("tags", tags), ("scenario", scenario),
                       ("custom_prompt", custom_prompt), ("camera_id", camera_id),
                       ("capture_type", capture_type), ("location", location)):
        if value:
            form[key] = value
    with open(path, "rb") as fh:
        files = {"file": (os.path.basename(path), fh, "video/mp4")}
        resp = requests.post(f"{backend()}/api/v1/videos/upload",
                             headers=_headers(), files=files, data=form, timeout=900)
    if resp.status_code == 401:
        login(force=True)
        with open(path, "rb") as fh:
            files = {"file": (os.path.basename(path), fh, "video/mp4")}
            resp = requests.post(f"{backend()}/api/v1/videos/upload",
                                 headers=_headers(), files=files, data=form, timeout=900)
    resp.raise_for_status()
    return resp.json()


def reingest(original_video: str = None, stream_id: str = None, chunk_count: int = 1,
             custom_prompt: str = None, scenario: str = None, **metadata) -> dict:
    payload = {"chunk_count": chunk_count}
    if stream_id:
        payload["stream_id"] = stream_id
    elif original_video:
        payload["original_video"] = original_video
    else:
        raise ValueError("reingest needs stream_id or original_video")
    if custom_prompt:
        payload["custom_prompt"] = custom_prompt
    elif scenario:
        payload["scenario"] = scenario
    payload.update({k: v for k, v in metadata.items() if v})
    return _request("POST", "/api/v1/dashboard/reingest", json=payload)


def reingest_status(job_id: str) -> dict:
    return _request("GET", f"/api/v1/dashboard/reingest/{job_id}")


def health() -> dict:
    """Credential-safe health summary for the README and the demo."""
    out = {"backend_reachable": False, "login": False, "indexed": None, "error": ""}
    try:
        login()
        out["login"] = True
        out["backend_reachable"] = True
        out["indexed"] = dashboard_stats().get("total_segments")
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


# The Tier-1 ingestion prompt. The pipeline writes one caption per 5s segment following
# this, and anything it does not ask about is not searchable later - so it names every
# football event we intend to retrieve. Max 800 characters.
FOOTBALL_INGEST_PROMPT = (
    "Describe this football (soccer) clip for search. State: which third of the pitch "
    "the ball is in; whether the ball is crossed from a wide area, played as a through "
    "ball behind defenders, struck as a shot, headed, or delivered from a corner, free "
    "kick, penalty or throw-in; whether a shot is on or off target, saved, blocked or "
    "scored; whether players celebrate, the goalkeeper retrieves the ball from the net, "
    "or the crowd reacts loudly; whether this is live action or a replay; and the camera "
    "type (wide pitch view, close-up on a player, crowd shot). Name shirt colours rather "
    "than teams. If little happens, say so plainly."
)
assert len(FOOTBALL_INGEST_PROMPT) <= 800, len(FOOTBALL_INGEST_PROMPT)
