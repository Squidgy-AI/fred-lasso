#!/usr/bin/env python3
"""Stitch the U-17 clip set into one continuous match video, and record the timeline.

The clips are published as separate passages of play. Treating them as one video gives
the anchor detector something match-shaped to work on, and the timeline map keeps every
event traceable back to the Commons file it came from.
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fred import clips  # noqa: E402

OUT = "data/raw/u17_nzl_can_match.mp4"
PARTS_DIR = "data/clips/u17"


def main():
    sources = sorted(glob.glob("data/raw/u17_nzl_can_[0-9][0-9].webm"))
    if not sources:
        print("No U-17 clips in data/raw. Run scripts/fetch_footage.py first.")
        return 1
    os.makedirs(PARTS_DIR, exist_ok=True)
    parts, offset, timeline = [], 0.0, []
    for index, src in enumerate(sources, 1):
        part = os.path.join(PARTS_DIR, f"part_{index:02d}.mp4")
        clips.cut(src, 0, clips.probe(src)["duration"], part, height=480)
        length = clips.probe(part)["duration"]
        timeline.append({"part": index, "src": os.path.basename(src),
                         "start": round(offset, 2), "end": round(offset + length, 2)})
        offset += length
        parts.append(part)
    clips.concat(parts, OUT)
    os.makedirs("data/out", exist_ok=True)
    with open("data/out/u17_timeline.json", "w") as fh:
        json.dump(timeline, fh, indent=2)
    print(f"{OUT}: {offset / 60:.1f} min from {len(parts)} clips")
    print("timeline: data/out/u17_timeline.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
