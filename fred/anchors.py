"""Anchor detection: find the moments worth asking Cosmos about.

A goal is the one football event that announces itself. The crowd noise floor jumps and
stays up for several seconds, which is a far stronger and cheaper signal than anything
visual - and critically it survives on footage with no scoreboard overlay, which is all
the footage we are allowed to show.

Three independent sources, each individually weak, jointly strong:
  crowd  - sustained audio energy above the local baseline
  asr    - NVIDIA Canary-1B transcribes the window and we keyword-match it
  caption- the Tier-1 Cosmos caption already in the index mentions a goal

Nothing here raises; a dead source returns an empty list and the others carry on.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass, field

import numpy as np
import requests

from . import config

SAMPLE_RATE = 16000
HOP_SECONDS = 1.0

GOAL_WORDS = (
    "goal", "scores", "scored", "scoring", "nets", "finish", "what a strike",
    "one nil", "two nil", "gol", "doelpunt",  # Dutch footage in the corpus
)


@dataclass
class Anchor:
    """A candidate goal moment in one source video."""

    t: float                       # peak of the reaction
    sources: list = field(default_factory=list)
    score: float = 0.0             # strength of the strongest source
    detail: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        return {
            "t_anchor": round(self.t, 2),
            "anchor_sources": ",".join(self.sources),
            "anchor_score": round(self.score, 3),
        }


def _decode_audio(path: str) -> np.ndarray:
    """Decode to mono 16kHz float32 in [-1, 1]. Empty array if there is no audio."""
    try:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-f", "s16le", "-"],
            capture_output=True, check=True,
        ).stdout
    except subprocess.CalledProcessError:
        return np.zeros(0, dtype=np.float32)
    if not raw:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def _rolling_median(values: np.ndarray, window: int) -> np.ndarray:
    """Local noise floor. A match has a loud first half and a flat second; a global
    threshold would over-fire on one and under-fire on the other."""
    if len(values) == 0:
        return values
    window = max(3, min(window, len(values)))
    pad = window // 2
    padded = np.pad(values, pad, mode="edge")
    strides = np.lib.stride_tricks.sliding_window_view(padded, window)
    return np.median(strides, axis=-1)[: len(values)]


def crowd_anchors(path: str, ratio: float = 1.55, min_sustain: float = 2.0,
                  merge_gap: float = 90.0, baseline_seconds: float = 180.0,
                  top_k: int = 0) -> list:
    """Sustained crowd-noise spikes above the rolling local baseline.

    ``min_sustain`` is what separates a goal from a single loud whistle or a camera bump:
    a goal reaction stays elevated for seconds, an impulse does not.
    """
    samples = _decode_audio(path)
    if samples.size < SAMPLE_RATE * 5:
        return []

    hop = int(SAMPLE_RATE * HOP_SECONDS)
    usable = (samples.size // hop) * hop
    frames = samples[:usable].reshape(-1, hop)
    rms = np.sqrt((frames ** 2).mean(axis=1)) + 1e-9

    # Adapt the baseline window to the clip: on a 4-minute highlights reel a 3-minute
    # window is the whole video, which flattens the very excursions we are looking for.
    effective_baseline = min(baseline_seconds, max(20.0, len(rms) * HOP_SECONDS / 4.0))
    baseline = _rolling_median(rms, int(effective_baseline / HOP_SECONDS))
    excess = rms / np.maximum(baseline, 1e-9)

    hot = excess >= ratio
    anchors, run_start = [], None
    for i, is_hot in enumerate(np.append(hot, False)):
        if is_hot and run_start is None:
            run_start = i
        elif not is_hot and run_start is not None:
            run_len = (i - run_start) * HOP_SECONDS
            if run_len >= min_sustain:
                segment = excess[run_start:i]
                peak_i = run_start + int(np.argmax(segment))
                anchors.append(Anchor(
                    t=peak_i * HOP_SECONDS,
                    sources=["crowd"],
                    score=float(segment.max()),
                    detail={"sustain_s": round(run_len, 1),
                            "onset_t": round(run_start * HOP_SECONDS, 1)},
                ))
            run_start = None

    anchors.sort(key=lambda a: a.t)
    merged = []
    for anchor in anchors:
        if merged and anchor.t - merged[-1].t < merge_gap:
            if anchor.score > merged[-1].score:   # keep the loudest of the cluster
                merged[-1] = anchor
            continue
        merged.append(anchor)

    if top_k:
        merged = sorted(merged, key=lambda a: a.score, reverse=True)[:top_k]
        merged.sort(key=lambda a: a.t)
    return merged


def transcribe(path: str, model: str = None) -> str:
    """NVIDIA Canary-1B ASR. Deployed on the challenge stack but not wired into the
    pipeline, so this is ours to add. Returns '' on any failure."""
    base = config.get("CANARY_1B_URL")
    if not base:
        return ""
    model = model or config.get("CANARY_1B_MODEL") or ""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", path, "-ac", "1", "-ar", "16000",
             wav_path],
            check=True, capture_output=True,
        )
        with open(wav_path, "rb") as fh:
            files = {"file": ("clip.wav", fh, "audio/wav")}
            data = {"model": model} if model else {}
            resp = requests.post(
                f"{base.rstrip('/')}/v1/audio/transcriptions",
                files=files, data=data, timeout=180,
            )
        resp.raise_for_status()
        payload = resp.json()
        return (payload.get("text") or "").strip()
    except Exception:  # noqa: BLE001 - optional source
        return ""
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)


def canary_health() -> tuple:
    base = config.get("CANARY_1B_URL")
    if not base:
        return False, "CANARY_1B_URL not set"
    try:
        resp = requests.get(f"{base.rstrip('/')}/v1/health/ready", timeout=30)
        return resp.ok, f"{resp.status_code} {resp.text[:80]}"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def mentions_goal(text: str) -> bool:
    low = (text or "").lower()
    return any(word in low for word in GOAL_WORDS)


def confirm_with_asr(src: str, anchors, pre: float = 12.0, post: float = 8.0) -> list:
    """Add an 'asr' source to any anchor whose audio actually says goal.

    Confirmation only: it never creates or drops an anchor, it raises confidence in one.
    """
    from . import clips  # local import keeps ffmpeg use out of module import time

    if not config.get("CANARY_1B_URL"):
        return anchors
    with tempfile.TemporaryDirectory() as tmpdir:
        for anchor in anchors:
            clip_path = os.path.join(tmpdir, f"asr_{int(anchor.t)}.mp4")
            try:
                clips.cut(src, anchor.t - pre, anchor.t + post, clip_path,
                          height=240, with_audio=True)
            except subprocess.CalledProcessError:
                continue
            text = transcribe(clip_path)
            if text:
                anchor.detail["asr_text"] = text[:300]
                if mentions_goal(text) and "asr" not in anchor.sources:
                    anchor.sources.append("asr")
    return anchors


def find_anchors(src: str, use_asr: bool = True, **kwargs) -> list:
    anchors = crowd_anchors(src, **kwargs)
    if use_asr and anchors:
        anchors = confirm_with_asr(src, anchors)
    return anchors
