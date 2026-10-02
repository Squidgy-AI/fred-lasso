#!/usr/bin/env python3
"""Hand-label the ground truth, one anchor at a time.

Ground truth is the only part of this project a model cannot produce. The eval is
worthless if the labels came from the same system being measured, so this plays you the
clip and records what you say, with a free-text note on what you based it on.

    python scripts/label.py                 # label anything still blank
    python scripts/label.py --relabel       # go through everything again
"""
import argparse, csv, os, subprocess, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fred import config, events, pipeline  # noqa: E402

LABELS = "data/labels/ground_truth.csv"
FIELDS = ["match", "t_anchor", "is_goal", "assist_type", "evidence", "labelled_by"]


def load():
    if not os.path.exists(LABELS):
        return []
    with open(LABELS, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save(rows):
    with open(LABELS, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})


def seed_from_events(rows):
    """Add a blank row for any anchor in the event table we have not seen before."""
    known = {(r["match"], str(int(float(r["t_anchor"])))) for r in rows}
    for event in events.query(limit=1000):
        key = (event["match"], str(int(event["t_anchor"])))
        if key not in known:
            rows.append({"match": event["match"], "t_anchor": key[1],
                         "is_goal": "", "assist_type": "", "evidence": "", "labelled_by": ""})
    return rows


def play(match, t_anchor):
    slug = pipeline.slug(match)
    path = f"data/clips/{slug}_{int(float(t_anchor)):05d}.mp4"
    if not os.path.exists(path):
        print(f"  (no clip at {path})")
        return
    opener = {"darwin": "open", "linux": "xdg-open"}.get(sys.platform, None)
    if opener:
        subprocess.run([opener, path], check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  clip: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--relabel", action="store_true")
    parser.add_argument("--who", default=os.environ.get("USER", "unknown"))
    args = parser.parse_args()

    rows = seed_from_events(load())
    todo = [r for r in rows if args.relabel or not r.get("is_goal")]
    if not todo:
        print("Everything is labelled. Run: python scripts/evaluate.py")
        return

    taxonomy = config.ASSIST_TYPES
    print(f"{len(todo)} anchor(s) to label. Enter to skip, q to stop and save.\n")
    for row in todo:
        minutes, seconds = divmod(int(float(row["t_anchor"])), 60)
        print(f"{row['match']} @ {minutes:02d}:{seconds:02d}")
        play(row["match"], row["t_anchor"])
        answer = input("  was a goal scored? [y/n/skip/q] ").strip().lower()
        if answer == "q":
            break
        if answer not in ("y", "n"):
            continue
        row["is_goal"] = "yes" if answer == "y" else "no"
        row["labelled_by"] = args.who
        if answer == "y":
            print("   " + "  ".join(f"{i+1}={t}" for i, t in enumerate(taxonomy)))
            choice = input("  final ball? [1-%d] " % len(taxonomy)).strip()
            if choice.isdigit() and 1 <= int(choice) <= len(taxonomy):
                row["assist_type"] = taxonomy[int(choice) - 1]
        row["evidence"] = input("  what did you base that on? ").strip()
        print()

    save(rows)
    done = sum(1 for r in rows if r.get("is_goal"))
    print(f"Saved {LABELS}: {done}/{len(rows)} labelled")


if __name__ == "__main__":
    main()
