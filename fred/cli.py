"""Fred Lasso command line. One entry point for the whole demo."""
from __future__ import annotations

import argparse
import json
import os
import sys

from . import agent, clips, events, pipeline, trace


def cmd_health(args):
    from scripts import health  # noqa: F401
    return 0


def cmd_build(args):
    trace.init()
    summaries = []
    for video in args.video:
        match = args.match or os.path.splitext(os.path.basename(video))[0]
        summary = pipeline.build_events(
            video, match=match, clip_dir=args.clip_dir, pre=args.pre, post=args.post,
            use_asr=not args.no_asr, use_cosmos=not args.no_cosmos,
            push_vastdb=args.vastdb, db_path=args.db,
            ratio=args.ratio, min_sustain=args.min_sustain, top_k=args.top_k,
        )
        summaries.append(summary)
        print(json.dumps(summary, indent=2))
    events.export_jsonl(db_path=args.db)
    print(f"\n{sum(s['events_written'] for s in summaries)} event(s) in {args.db or events.LOCAL_DB}")
    return 0


def cmd_ask(args):
    trace.init()
    result = agent.ask(args.question, db_path=args.db)
    spec = result["filter"]
    print(f"\nQ: {result['question']}")
    print(f"compiled filter: assist_types={spec.get('assist_types')} "
          f"match={spec.get('match')} min_confidence={spec.get('min_confidence')}"
          + ("  [keyword fallback]" if spec.get("_fallback") else ""))
    if spec.get("explanation"):
        print(f"   reading: {spec['explanation']}")
    print(f"\n{result['answer']}\n")
    for row in result["events"]:
        minutes, seconds = divmod(int(row["t_anchor"]), 60)
        print(f"  {row['match'][:30]:30s} {minutes:02d}:{seconds:02d}  "
              f"{row['assist_type'] or '-':16s} conf={row['confidence']:.2f}  "
              f"[{row['anchor_sources']}]  {row['clip_path']}")
    if result.get("semantic_hits"):
        print("\n  (fell back to semantic search over the pre-built index)")
        for hit in result["semantic_hits"][:5]:
            print(f"    {hit.get('similarity_score', 0):.3f}  {hit.get('source', '')}")
    if args.reel and result["events"]:
        path = pipeline.highlights_reel(result["events"], out_path=args.reel)
        print(f"\nreel: {path}")
    return 0


def cmd_reel(args):
    rows = events.query(
        where="assist_type = ?" if args.assist_type else "",
        params=(args.assist_type,) if args.assist_type else (),
        limit=args.limit, path=args.db,
    )
    if not rows:
        print("no matching events")
        return 1
    print(pipeline.highlights_reel(rows, out_path=args.out))
    return 0


def cmd_stats(args):
    print(json.dumps(events.stats(path=args.db), indent=2))
    return 0


def cmd_serve(args):
    from app.server import serve
    serve(host=args.host, port=args.port, db_path=args.db)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="fred", description="Fred Lasso - a football video analyst agent")
    parser.add_argument("--db", default=None, help="SQLite event table path")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("health", help="check every stack component")
    p.set_defaults(func=cmd_health)

    p = sub.add_parser("build", help="video -> anchors -> Cosmos -> event table")
    p.add_argument("video", nargs="+")
    p.add_argument("--match", default=None, help="match name (default: filename)")
    p.add_argument("--clip-dir", default="data/clips")
    p.add_argument("--pre", type=float, default=clips.DEFAULT_PRE_ROLL,
                   help="seconds of build-up before the anchor")
    p.add_argument("--post", type=float, default=clips.DEFAULT_POST_ROLL)
    p.add_argument("--ratio", type=float, default=1.55, help="crowd spike threshold")
    p.add_argument("--min-sustain", type=float, default=2.0)
    p.add_argument("--top-k", type=int, default=0, help="keep only the N loudest anchors")
    p.add_argument("--no-asr", action="store_true", help="skip Canary-1B confirmation")
    p.add_argument("--no-cosmos", action="store_true", help="anchors only, no classification")
    p.add_argument("--vastdb", action="store_true", help="also write rows to VastDB")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("ask", help="ask a plain-language question")
    p.add_argument("question")
    p.add_argument("--reel", default=None, help="also cut a reel to this path")
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("reel", help="cut a highlights reel from the event table")
    p.add_argument("--assist-type", default=None)
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--out", default="data/out/highlights.mp4")
    p.set_defaults(func=cmd_reel)

    p = sub.add_parser("stats", help="what is in the event table")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("serve", help="run the demo UI")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8800)
    p.set_defaults(func=cmd_serve)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
