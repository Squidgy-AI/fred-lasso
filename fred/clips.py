"""ffmpeg surgery: probe, cut a build-up window, and concatenate a reel.

Everything Cosmos sees is produced here. We transcode to H.264/MP4 at a modest
resolution because the source footage is webm/ogv and because the quality ceiling on a
Cosmos Reason call is a multimodal token budget, not pixels.
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass

DEFAULT_PRE_ROLL = 20.0   # seconds of build-up before the anchor
DEFAULT_POST_ROLL = 3.0   # a little after, so the goal itself is visible


@dataclass
class Window:
    start: float
    end: float
    path: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def probe(path: str) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", path],
        capture_output=True, text=True, check=True,
    ).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), {})
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), {})
    return {
        "duration": float(data["format"].get("duration", 0.0)),
        "size": int(data["format"].get("size", 0)),
        "width": video.get("width"),
        "height": video.get("height"),
        "video_codec": video.get("codec_name"),
        "audio_codec": audio.get("codec_name"),
        "has_audio": bool(audio),
    }


def cut(src: str, start: float, end: float, out_path: str,
        height: int = 480, with_audio: bool = True) -> Window:
    """Cut [start, end) to MP4. Seeks before -i for speed, then re-encodes for accuracy."""
    start = max(0.0, start)
    duration = max(0.1, end - start)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-ss", f"{start:.3f}", "-i", src, "-t", f"{duration:.3f}",
        "-vf", f"scale=-2:{height}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
    ]
    cmd += (["-c:a", "aac", "-b:a", "96k"] if with_audio else ["-an"])
    cmd += [out_path]
    subprocess.run(cmd, check=True, capture_output=True)
    return Window(start=start, end=start + duration, path=out_path)


def build_up_window(src: str, anchor_t: float, out_path: str,
                    pre: float = DEFAULT_PRE_ROLL, post: float = DEFAULT_POST_ROLL,
                    height: int = 480) -> Window:
    """The clip the index never saw: the seconds leading into the anchor."""
    return cut(src, anchor_t - pre, anchor_t + post, out_path, height=height)


def concat(clip_paths, out_path: str) -> str:
    """Stitch clips into one reel. Re-encodes, so mismatched inputs are fine."""
    if not clip_paths:
        raise ValueError("concat needs at least one clip")
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    list_path = out_path + ".txt"
    with open(list_path, "w", encoding="utf-8") as fh:
        for path in clip_paths:
            fh.write(f"file '{os.path.abspath(path)}'\n")
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", list_path,
         "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out_path],
        check=True, capture_output=True,
    )
    os.remove(list_path)
    return out_path


def split_for_upload(src: str, out_dir: str, max_mb: int = 24,
                     height: int = 480, crf: int = 30) -> list:
    """Transcode and split into parts that fit the backend's upload cap.

    The VSS upload endpoint takes one file per request with ``max_upload_size_mb``
    typically 25-100MB, so a full match has to arrive as several parts. Bitrate is
    chosen from the cap rather than guessed, then each part is cut to a fixed duration.
    """
    os.makedirs(out_dir, exist_ok=True)
    meta = probe(src)
    duration = meta["duration"]
    # Budget ~85% of the cap for video, leaving room for audio and container overhead.
    target_kbps = 700
    seconds_per_part = max(30.0, (max_mb * 8192 * 0.85) / target_kbps)
    parts, index, start = [], 0, 0.0
    while start < duration:
        part_path = os.path.join(out_dir, f"part_{index:03d}.mp4")
        length = min(seconds_per_part, duration - start)
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", src,
             "-t", f"{length:.3f}", "-vf", f"scale=-2:{height}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
             "-maxrate", f"{target_kbps}k", "-bufsize", f"{target_kbps * 2}k",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k",
             "-movflags", "+faststart", part_path],
            check=True, capture_output=True,
        )
        size_mb = os.path.getsize(part_path) / 1e6
        parts.append({"path": part_path, "start": start, "end": start + length,
                      "size_mb": round(size_mb, 1)})
        start += length
        index += 1
    return parts
