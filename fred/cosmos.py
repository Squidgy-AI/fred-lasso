"""The constrained second pass: ask Cosmos3-Reason a multiple-choice question.

The provided pipeline already writes a free-text caption per 5-second segment. That is
good for recall and bad for a compositional question, because a caption written without
being asked about the final ball cannot be filtered on the final ball.

So we re-ask. We stitch the build-up window the index never saw as a single clip and put
a fixed, closed set of options in front of the model. Constraining the output space is
what buys accuracy here: the model picks a letter instead of inventing a description.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
from dataclasses import dataclass, field

import requests

from . import config

TIMEOUT = int(os.environ.get("FRED_COSMOS_TIMEOUT", "600"))  # blueprint uses 600s

# Asking for <think> then <answer> is NVIDIA's own recommended pattern for the Cosmos
# Reason family, and it is what makes the letter parse reliably.
_ASSIST_PROMPT = """You are analysing a short clip of association football (soccer).
The clip was selected because the crowd reacted loudly near the end of it, which often
but not always means a goal was scored.

Answer two questions.

QUESTION 1 - was a goal actually scored in this clip? Answer YES only if you can see the
ball enter the net, players celebrating a goal, or the goalkeeper retrieving the ball
from inside the net. Loud noise alone is not a goal: a near miss, a save, a foul, a
substitution or a crowd chant are all NO.

QUESTION 2 - if and only if a goal was scored, what kind of final ball - the last
meaningful pass, delivery or action before the shot - created it?

Choose exactly one option:
{options}

Definitions:
A cross from wide is a ball delivered from a wide area towards the penalty box.
A through ball is a pass played between or behind defenders into space.
A set piece is a goal from a corner, free kick, penalty or throw-in routine.
A long-range shot is a shot struck from outside the penalty area with no final pass.
A rebound is a goal scored from a save, block, deflection or loose ball in the box.
Choose other only if none of the above fits.

Answer in exactly this format:
<think>
your reasoning about what you can actually see
</think>

<answer>
GOAL=YES|LETTER|CONFIDENCE
</answer>

or, when no goal was scored:

<answer>
GOAL=NO|F|CONFIDENCE
</answer>

where LETTER is one of {letters} and CONFIDENCE is a number from 0.0 to 1.0 describing
how certain you are. If the camera is too far away or the action is unclear, say so in
your reasoning and give a low confidence."""


@dataclass
class AssistVerdict:
    """What Cosmos decided about one build-up window."""

    assist_type: str
    confidence: float
    letter: str
    reasoning: str
    raw: str
    model: str
    goal_confirmed: bool = True
    ok: bool = True
    error: str = ""
    options: list = field(default_factory=lambda: list(config.ASSIST_TYPES))

    def as_row(self) -> dict:
        return {
            "goal_confirmed": self.goal_confirmed,
            "assist_type": self.assist_type,
            "confidence": round(self.confidence, 3),
            "cosmos_letter": self.letter,
            "cosmos_reasoning": self.reasoning,
            "cosmos_model": self.model,
            "cosmos_ok": self.ok,
            "cosmos_error": self.error,
        }


def _base_url() -> str:
    return config.require("COSMOS3_REASON_URL").rstrip("/")


def _model() -> str:
    return config.get("COSMOS3_REASON_MODEL") or "nvidia/cosmos3-reason"


def _letters(n: int) -> list:
    return [chr(ord("A") + i) for i in range(n)]


def _format_options(options) -> str:
    return "\n".join(
        f"{letter}. {name}" for letter, name in zip(_letters(len(options)), options)
    )


def health() -> tuple:
    """Smoke test matching the gpu/model-smoke-test skill."""
    try:
        resp = requests.post(
            f"{_base_url()}/v1/chat/completions",
            json={
                "model": _model(),
                "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        return bool(content and content.strip()), (content or "").strip()
    except Exception as exc:  # noqa: BLE001 - health check reports, never raises
        return False, f"{type(exc).__name__}: {exc}"


def _video_data_uri(path: str) -> str:
    with open(path, "rb") as fh:
        return "data:video/mp4;base64," + base64.b64encode(fh.read()).decode("ascii")


def _sample_frames(path: str, fps: float = 1.0, max_frames: int = 16) -> list:
    """Fall back to JPEG frames when a whole-video payload is refused.

    NVIDIA documents ``video_frames`` as preserving temporal order at a lower token cost
    than the same images sent as separate image_url blocks, so this is a real fallback
    rather than a degraded one.
    """
    cmd = [
        "ffmpeg", "-v", "error", "-i", path,
        "-vf", f"fps={fps},scale=448:-2",
        "-frames:v", str(max_frames), "-f", "image2pipe", "-vcodec", "mjpeg", "-",
    ]
    blob = subprocess.run(cmd, capture_output=True, check=True).stdout
    # Split the concatenated MJPEG stream on JPEG SOI/EOI markers.
    frames, start = [], 0
    while True:
        soi = blob.find(b"\xff\xd8", start)
        if soi < 0:
            break
        eoi = blob.find(b"\xff\xd9", soi)
        if eoi < 0:
            break
        frames.append(base64.b64encode(blob[soi:eoi + 2]).decode("ascii"))
        start = eoi + 2
    return frames


def _parse_answer(text: str, options) -> tuple:
    """Pull GOAL=YES/NO|LETTER|CONFIDENCE out of the <answer> block, tolerantly."""
    letters = _letters(len(options))
    answer_block = ""
    match = re.search(r"<answer>(.*?)</answer>", text, re.S | re.I)
    if match:
        answer_block = match.group(1).strip()
    else:
        # The model sometimes opens <answer> and never closes it; take the tail.
        tail = re.split(r"<answer>", text, flags=re.I)
        answer_block = tail[-1].strip() if len(tail) > 1 else text.strip()

    upper = answer_block.upper()
    goal_confirmed = True
    goal_match = re.search(r"GOAL\s*=\s*(YES|NO)", upper)
    if goal_match:
        goal_confirmed = goal_match.group(1) == "YES"
    # Strip the GOAL= clause before hunting for the option letter, or the Y/N trips it.
    letter_zone = re.sub(r"GOAL\s*=\s*(YES|NO)", "", upper)
    letter_match = re.search(r"\b([A-%s])\b" % letters[-1], letter_zone)
    letter = letter_match.group(1) if letter_match else ""

    conf = 0.0
    conf_match = re.search(r"(\d*\.?\d+)\s*%?", answer_block.split("|")[-1])
    if conf_match:
        conf = float(conf_match.group(1))
        if conf > 1.0:  # the model sometimes answers in percent
            conf = conf / 100.0
    conf = max(0.0, min(1.0, conf))

    assist = options[letters.index(letter)] if letter in letters else "other"
    if not goal_confirmed:
        assist = ""
    reasoning = ""
    think = re.search(r"<think>(.*?)</think>", text, re.S | re.I)
    if think:
        reasoning = " ".join(think.group(1).split())
    return assist, conf, letter, reasoning, goal_confirmed


def classify_assist(clip_path: str, options=None, max_tokens: int = 4096) -> AssistVerdict:
    """Classify the final ball in one build-up clip. Never raises."""
    options = list(options or config.ASSIST_TYPES)
    prompt = _ASSIST_PROMPT.format(
        options=_format_options(options),
        letters="/".join(_letters(len(options))),
    )
    model = _model()

    def _post(content) -> str:
        resp = requests.post(
            f"{_base_url()}/v1/chat/completions",
            json={
                "model": model,
                "messages": [{"role": "user", "content": content}],
                # Low but non-zero: greedy decoding on this family tends to loop in <think>.
                "temperature": 0.2,
                "top_p": 0.3,
                "max_tokens": max_tokens,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""

    attempts = [
        ("video_url", lambda: [
            {"type": "video_url", "video_url": {"url": _video_data_uri(clip_path)}},
            {"type": "text", "text": prompt},
        ]),
        ("video_frames", lambda: [
            {"type": "video_frames", "video_frames": _sample_frames(clip_path)},
            {"type": "text", "text": prompt},
        ]),
    ]

    last_error = ""
    for label, build in attempts:
        try:
            raw = _post(build())
            if not raw.strip():
                last_error = f"{label}: empty content (model overloaded or input rejected)"
                continue
            assist, conf, letter, reasoning, confirmed = _parse_answer(raw, options)
            return AssistVerdict(
                assist_type=assist, confidence=conf, letter=letter,
                reasoning=reasoning, raw=raw, model=model, options=options,
                goal_confirmed=confirmed,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = f"{label}: {type(exc).__name__}: {exc}"

    return AssistVerdict(
        assist_type="other", confidence=0.0, letter="", reasoning="", raw="",
        model=model, ok=False, error=last_error, options=options,
        goal_confirmed=False,
    )


def describe_clip(clip_path: str, question: str, max_tokens: int = 1024) -> str:
    """Free-text question about a clip. Used for the unconstrained ablation."""
    try:
        resp = requests.post(
            f"{_base_url()}/v1/chat/completions",
            json={
                "model": _model(),
                "messages": [{"role": "user", "content": [
                    {"type": "video_url", "video_url": {"url": _video_data_uri(clip_path)}},
                    {"type": "text", "text": question},
                ]}],
                "temperature": 0.2, "max_tokens": max_tokens,
            },
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""
    except Exception as exc:  # noqa: BLE001
        return f"ERROR {type(exc).__name__}: {exc}"


if __name__ == "__main__":
    ok, detail = health()
    print(json.dumps({"cosmos_reason_healthy": ok, "detail": detail}, indent=2))
