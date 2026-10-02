#!/usr/bin/env python3
"""Run the ablation evaluation and print the report."""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fred import evaluate  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--db", default=None)
parser.add_argument("--labels", default=evaluate.LABELS_PATH)
parser.add_argument("--with-semantic", action="store_true",
                    help="also run the hybrid-search baseline (needs the VSS backend)")
parser.add_argument("--no-weave", action="store_true")
parser.add_argument("--json", action="store_true")
args = parser.parse_args()

summary = evaluate.run(db_path=args.db, labels_path=args.labels,
                       with_semantic=args.with_semantic, log_weave=not args.no_weave)
print(json.dumps(summary, indent=2) if args.json else evaluate.format_report(summary))
os.makedirs("data/out", exist_ok=True)
with open("data/out/eval_report.json", "w") as fh:
    json.dump(summary, fh, indent=2)
