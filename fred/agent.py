"""The agent: turn a plain-language football question into a filter over the event table.

The judgement call here is that most useful football questions are *structured* even
when they are phrased casually. "Crosses that led to a goal" is a filter, not a vibe.
So the LLM's job is not to answer the question - it is to compile the question into a
filter, which we then execute deterministically against the event table. That keeps the
answer grounded in rows we verified with Cosmos, and it makes the agent explainable:
we can show the filter it produced.

Questions that genuinely are fuzzy ("scrappy goals", "chaotic moments") fall through to
hybrid semantic search over the pre-built index instead.

LLM inference runs on Weights & Biases serverless inference (a CoreWeave company),
which is the sponsor path for an app's own reasoning.
"""
from __future__ import annotations

import json
import os
import re

import requests

from . import config, events

WANDB_BASE_URL = os.environ.get("WANDB_BASE_URL", "https://api.inference.wandb.ai/v1")
PREFERRED_MODELS = [
    "meta-llama/Llama-3.3-70B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "deepseek-ai/DeepSeek-V3.1",
    "Qwen/Qwen3-235B-A22B-Instruct-2507",
]

FILTER_SCHEMA_HINT = {
    "mode": "structured | semantic",
    "event_type": "goal",
    "assist_types": ["subset of the taxonomy, empty means any"],
    "match": "substring of a match name, or null",
    "min_confidence": 0.0,
    "limit": 10,
    "explanation": "one short sentence on how you read the question",
}

SYSTEM_PROMPT = """You convert football questions into a JSON filter over a table of
verified goal events.

Each row has: match, t_anchor (seconds into the video), event_type (always "goal"),
assist_type (exactly one of: {taxonomy}), confidence (0-1, how sure the vision model was),
anchor_sources (how the goal was detected), clip_path.

Return ONLY a JSON object with these keys:
  mode           "structured" if the question maps onto assist_type/match/confidence
                 filters; "semantic" if it asks about something not in the table
                 (mood, weather, crowd, tactics, a named player).
  event_type     always "goal" for now.
  assist_types   list, a subset of the taxonomy. Empty list means any.
  match          substring to match against the match name, or null.
  min_confidence number 0-1. Use 0 unless the user asks for confident/certain results.
  limit          integer, default 10.
  explanation    one short sentence describing how you read the question.

Map everyday football language onto the taxonomy. "Crosses", "balls in from the wing",
"deliveries from wide" -> "cross from wide". "Through balls", "slipped in", "balls in
behind" -> "through ball". "Set pieces", "corners", "free kicks", "penalties" -> "set
piece". "Screamers", "worldies", "from distance", "outside the box" -> "long-range shot".
"Scrappy", "rebounds", "second balls", "tap-ins from a save" -> "rebound".

No prose, no markdown fences, JSON only."""


class InferenceError(RuntimeError):
    pass


def api_key() -> str:
    return config.require("WANDB_API_KEY")


def project() -> str:
    team = config.get("WANDB_TEAM")
    proj = config.get("WANDB_PROJECT") or "fred-lasso"
    return f"{team}/{proj}" if team else proj


def list_models() -> list:
    try:
        resp = requests.get(
            f"{WANDB_BASE_URL}/models",
            headers={"Authorization": f"Bearer {api_key()}"}, timeout=60,
        )
        resp.raise_for_status()
        return [m["id"] for m in resp.json().get("data", [])]
    except Exception:  # noqa: BLE001
        return []


def pick_model() -> str:
    """Honour an explicit choice, else the first preferred model the catalogue serves."""
    explicit = os.environ.get("FRED_LLM_MODEL")
    if explicit:
        return explicit
    available = list_models()
    for candidate in PREFERRED_MODELS:
        if candidate in available:
            return candidate
    return available[0] if available else PREFERRED_MODELS[-1]


def chat(messages, model: str = None, temperature: float = 0.0,
         max_tokens: int = 800) -> str:
    model = model or pick_model()
    headers = {"Authorization": f"Bearer {api_key()}", "Content-Type": "application/json"}
    team_project = project()
    if team_project:
        headers["OpenAI-Project"] = team_project
    resp = requests.post(
        f"{WANDB_BASE_URL}/chat/completions",
        headers=headers,
        json={"model": model, "messages": messages,
              "temperature": temperature, "max_tokens": max_tokens},
        timeout=180,
    )
    if not resp.ok:
        raise InferenceError(f"{resp.status_code}: {resp.text[:300]}")
    return resp.json()["choices"][0]["message"]["content"] or ""


def _extract_json(text: str) -> dict:
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError(f"no JSON object in model output: {text[:200]}")
    return json.loads(match.group(0))


def _fallback_filter(question: str) -> dict:
    """Keyword compiler used when inference is unavailable, so the demo still runs."""
    low = question.lower()
    buckets = {
        "cross from wide": ("cross", "crosses", "wing", "wide", "delivery", "whipped"),
        "through ball": ("through ball", "through-ball", "slipped", "in behind", "played in"),
        "set piece": ("set piece", "corner", "free kick", "free-kick", "penalty", "throw"),
        "long-range shot": ("long range", "long-range", "distance", "outside the box",
                            "screamer", "worldie"),
        "rebound": ("rebound", "scrappy", "second ball", "loose ball", "tap-in"),
    }
    chosen = [name for name, words in buckets.items() if any(w in low for w in words)]
    return {
        "mode": "structured", "event_type": "goal", "assist_types": chosen,
        "match": None, "min_confidence": 0.0, "limit": 10,
        "explanation": "keyword fallback (W&B inference unavailable)",
        "_fallback": True,
    }


def compile_query(question: str, model: str = None) -> dict:
    """Question -> filter spec. Falls back to keywords rather than failing the demo."""
    system = SYSTEM_PROMPT.format(taxonomy=", ".join(config.ASSIST_TYPES))
    try:
        raw = chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": question}],
            model=model, temperature=0.0, max_tokens=400,
        )
        spec = _extract_json(raw)
    except Exception as exc:  # noqa: BLE001
        spec = _fallback_filter(question)
        spec["_error"] = f"{type(exc).__name__}: {exc}"
        return spec

    valid = set(config.ASSIST_TYPES)
    spec.setdefault("mode", "structured")
    spec["assist_types"] = [a for a in (spec.get("assist_types") or []) if a in valid]
    spec["match"] = spec.get("match") or None
    try:
        spec["min_confidence"] = float(spec.get("min_confidence") or 0.0)
    except (TypeError, ValueError):
        spec["min_confidence"] = 0.0
    try:
        spec["limit"] = max(1, min(100, int(spec.get("limit") or 10)))
    except (TypeError, ValueError):
        spec["limit"] = 10
    return spec


def run_filter(spec: dict, db_path: str = None) -> list:
    """Execute the compiled filter against the event table. Deterministic, no model."""
    # Only Cosmos-confirmed goals are answers. Rejected anchors stay in the table as
    # evidence of what the audio alone would have returned, which is the eval story.
    where = ["event_type = ?", "goal_confirmed = 1"]
    params = [spec.get("event_type") or "goal"]
    assist_types = spec.get("assist_types") or []
    if assist_types:
        where.append("assist_type IN (%s)" % ",".join("?" for _ in assist_types))
        params.extend(assist_types)
    if spec.get("match"):
        where.append("match LIKE ?")
        params.append(f"%{spec['match']}%")
    if spec.get("min_confidence"):
        where.append("confidence >= ?")
        params.append(float(spec["min_confidence"]))
    return events.query(
        where=" AND ".join(where), params=tuple(params),
        order="confidence DESC, match, t_anchor",
        limit=spec.get("limit", 10), path=db_path,
    )


def summarise(question: str, rows: list, model: str = None) -> str:
    """One short analyst's answer over the rows we actually found. Never invents rows."""
    if not rows:
        return "No events in the table match that question."
    facts = [
        f"- {r['match']} at {int(r['t_anchor']) // 60:02d}:{int(r['t_anchor']) % 60:02d}"
        f" | {r['assist_type']} (confidence {r['confidence']:.2f})"
        f" | detected by {r['anchor_sources']}"
        for r in rows
    ]
    try:
        return chat(
            [{"role": "system", "content":
              "You are Fred Lasso, a concise football analyst. Answer the user's question "
              "using ONLY the listed events. Two or three sentences. Mention how many "
              "there were and any pattern across them. Never invent an event."},
             {"role": "user", "content":
              f"Question: {question}\n\nEvents found:\n" + "\n".join(facts)}],
            model=model, temperature=0.3, max_tokens=250,
        ).strip()
    except Exception:  # noqa: BLE001
        kinds = {}
        for r in rows:
            kinds[r["assist_type"]] = kinds.get(r["assist_type"], 0) + 1
        breakdown = ", ".join(f"{v} x {k}" for k, v in kinds.items())
        return f"Found {len(rows)} matching goal(s): {breakdown}."


def ask(question: str, db_path: str = None, model: str = None,
        semantic_fallback: bool = True) -> dict:
    """The whole agent loop: compile, execute, summarise, return clips as evidence."""
    spec = compile_query(question, model=model)
    rows = run_filter(spec, db_path=db_path)

    used_semantic = False
    if not rows and semantic_fallback:
        try:
            from . import vss
            hits = vss.search(question, top_k=10, min_similarity=0.25)
            used_semantic = True
            return {
                "question": question, "filter": spec, "events": [],
                "semantic_hits": hits.get("results", [])[:10],
                "answer": (hits.get("llm_synthesis") or {}).get(
                    "response", "No structured events matched; showing semantic hits."),
                "used_semantic_fallback": True,
            }
        except Exception:  # noqa: BLE001
            pass

    return {
        "question": question,
        "filter": spec,
        "events": rows,
        "semantic_hits": [],
        "answer": summarise(question, rows, model=model),
        "used_semantic_fallback": used_semantic,
    }
