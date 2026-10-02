"""End-to-end: a source video in, verified goal events out.

    anchors -> build-up windows -> constrained Cosmos -> event table

This is the loop the whole project exists to demonstrate. Tier 1 (the pre-built index)
is good at recall over hours; it cannot answer a compositional question. So we find
candidate moments cheaply, cut the window that precedes each one, and put a closed
multiple-choice question to Cosmos about a clip the index never saw as a unit.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

from . import anchors as anchors_mod
from . import clips, cosmos, events, trace


def _event_id(match: str, t: float) -> str:
    return hashlib.sha1(f"{match}|{t:.1f}".encode("utf-8")).hexdigest()[:16]


def slug(text: str) -> str:
    """Filesystem- and URL-safe name. Clip paths end up in ffmpeg concat lists and in
    the web UI, and spaces break both in different ways."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "match"


@trace.op
def classify_window(clip_path: str, match: str, t_anchor: float) -> dict:
    """One constrained Cosmos call. Traced so the clip and reasoning are inspectable."""
    verdict = cosmos.classify_assist(clip_path)
    return {
        "match": match, "t_anchor": t_anchor, "clip_path": clip_path,
        **verdict.as_row(),
    }


@trace.op
def detect_anchors(video: str, use_asr: bool = True, **kwargs) -> list:
    found = anchors_mod.find_anchors(video, use_asr=use_asr, **kwargs)
    return [{"t": a.t, "sources": a.sources, "score": a.score, **a.detail} for a in found]


def build_events(video: str, match: str = None, clip_dir: str = "data/clips",
                 pre: float = clips.DEFAULT_PRE_ROLL,
                 post: float = clips.DEFAULT_POST_ROLL,
                 use_asr: bool = True, use_cosmos: bool = True,
                 push_vastdb: bool = False, db_path: str = None,
                 progress=print, **anchor_kwargs) -> dict:
    """Run the whole loop over one video and persist the rows."""
    match = match or os.path.splitext(os.path.basename(video))[0]
    os.makedirs(clip_dir, exist_ok=True)
    started = time.time()

    meta = clips.probe(video)
    progress(f"[{match}] {meta['duration'] / 60:.1f} min, scanning audio for anchors...")
    found = anchors_mod.find_anchors(video, use_asr=use_asr, **anchor_kwargs)
    progress(f"[{match}] {len(found)} candidate anchor(s)")

    rows, failures = [], 0
    for index, anchor in enumerate(found, 1):
        clip_path = os.path.join(clip_dir, f"{slug(match)}_{int(anchor.t):05d}.mp4")
        try:
            window = clips.build_up_window(video, anchor.t, clip_path, pre=pre, post=post)
        except Exception as exc:  # noqa: BLE001
            progress(f"  [{index}/{len(found)}] {anchor.t:.0f}s clip failed: {exc}")
            failures += 1
            continue

        verdict_row = {"goal_confirmed": True, "assist_type": "", "confidence": 0.0, "cosmos_letter": "",
                       "cosmos_reasoning": "", "cosmos_model": "", "cosmos_ok": 1,
                       "cosmos_error": ""}
        if use_cosmos:
            t0 = time.time()
            verdict_row = classify_window(clip_path, match, anchor.t)
            took = time.time() - t0
            confirmed = verdict_row.get("goal_confirmed", True)
            label = (verdict_row.get("assist_type") or "?") if confirmed else "NOT A GOAL"
            conf = verdict_row.get("confidence") or 0.0
            ok = verdict_row.get("cosmos_ok")
            progress(f"  [{index}/{len(found)}] {int(anchor.t)//60:02d}:{int(anchor.t)%60:02d} "
                     f"-> {label} ({conf:.2f}) in {took:.1f}s"
                     + ("" if ok else f"  FAILED: {verdict_row.get('cosmos_error','')[:90]}"))
            if not ok:
                failures += 1

        event = events.Event(
            event_id=_event_id(match, anchor.t),
            match=match,
            source_video=os.path.abspath(video),
            event_type="goal",
            t_anchor=anchor.t,
            window_start=window.start,
            window_end=window.end,
            goal_confirmed=1 if verdict_row.get("goal_confirmed", True) else 0,
            assist_type=verdict_row.get("assist_type", ""),
            confidence=float(verdict_row.get("confidence") or 0.0),
            anchor_sources=",".join(anchor.sources),
            anchor_score=anchor.score,
            cosmos_letter=verdict_row.get("cosmos_letter", ""),
            cosmos_reasoning=verdict_row.get("cosmos_reasoning", ""),
            cosmos_model=verdict_row.get("cosmos_model", ""),
            cosmos_ok=1 if verdict_row.get("cosmos_ok", True) else 0,
            cosmos_error=verdict_row.get("cosmos_error", ""),
            clip_path=clip_path,
            asr_text=anchor.detail.get("asr_text", ""),
        )
        rows.append(event)

    if rows:
        events.upsert(rows, path=db_path)

    pushed = {"ok": False, "written": 0, "error": "not attempted"}
    if push_vastdb and rows:
        pushed = events.push_to_vastdb(rows)
        progress(f"[{match}] VastDB: {pushed}")

    return {
        "match": match,
        "video": video,
        "duration_min": round(meta["duration"] / 60, 1),
        "anchors": len(found),
        "events_written": len(rows),
        "cosmos_failures": failures,
        "vastdb": pushed,
        "elapsed_s": round(time.time() - started, 1),
    }


def highlights_reel(rows, out_path: str = "data/out/highlights.mp4") -> str:
    """The 'act' step: cut the matching clips into one reel."""
    paths = [r["clip_path"] for r in rows if r.get("clip_path") and os.path.exists(r["clip_path"])]
    if not paths:
        raise ValueError("no clips available for a reel")
    return clips.concat(paths, out_path)
