"""Evaluation: does the decomposition actually earn its complexity?

The claim this project makes is that a compositional question needs more than one
embedding search. That is only worth saying if it is measured, so we run the same
labelled set through progressively more of the system and report what each stage buys:

  A  anchors only        every crowd spike is called a goal        -> recall, poor precision
  B  anchors + Cosmos    Cosmos vetoes the spikes that are not goals -> precision
  C  B + constrained MCQ assist type from the fixed taxonomy       -> accuracy on the real question
  D  semantic baseline   the question straight into hybrid search  -> what we are beating

Metrics are computed here and logged to Weave so the four runs sit side by side.
"""
from __future__ import annotations

import csv
import json
import os

from . import config, events, trace

LABELS_PATH = "data/labels/ground_truth.csv"
MATCH_TOLERANCE = 15.0  # seconds; an anchor within this of a labelled goal is that goal


def load_labels(path: str = LABELS_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("is_goal") or "").strip()]
    for row in rows:
        row["t_anchor"] = float(row["t_anchor"])
        row["is_goal"] = row["is_goal"].strip().lower() in ("yes", "y", "true", "1")
    return rows


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3),
            "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}


def _pair(label, rows):
    """The event row for a labelled anchor, if one exists within tolerance."""
    best, best_gap = None, MATCH_TOLERANCE
    for row in rows:
        if row["match"] != label["match"]:
            continue
        gap = abs(row["t_anchor"] - label["t_anchor"])
        if gap <= best_gap:
            best, best_gap = row, gap
    return best


def confusion(pairs) -> dict:
    """assist_type confusion, predicted vs labelled, over true goals only."""
    table = {}
    for truth, predicted in pairs:
        table.setdefault(truth, {}).setdefault(predicted, 0)
        table[truth][predicted] += 1
    return table


def run(db_path: str = None, labels_path: str = LABELS_PATH,
        with_semantic: bool = False, log_weave: bool = True) -> dict:
    labels = load_labels(labels_path)
    rows = events.query(limit=1000, path=db_path)
    if not labels:
        return {"error": f"no labels in {labels_path} - run scripts/label.py first"}

    results = {}

    # --- A: anchors only. Every anchor the audio proposed is called a goal.
    tp = fp = fn = 0
    for label in labels:
        row = _pair(label, rows)
        if row and label["is_goal"]:
            tp += 1
        elif row and not label["is_goal"]:
            fp += 1
        elif not row and label["is_goal"]:
            fn += 1
    results["A_anchors_only"] = _prf(tp, fp, fn)

    # --- B: anchors + Cosmos goal gate.
    tp = fp = fn = 0
    for label in labels:
        row = _pair(label, rows)
        predicted_goal = bool(row) and bool(row.get("goal_confirmed", 1))
        if predicted_goal and label["is_goal"]:
            tp += 1
        elif predicted_goal and not label["is_goal"]:
            fp += 1
        elif not predicted_goal and label["is_goal"]:
            fn += 1
    results["B_anchors_plus_cosmos_gate"] = _prf(tp, fp, fn)

    # --- C: assist-type accuracy over goals both we and the labeller call goals.
    pairs, correct = [], 0
    for label in labels:
        if not label["is_goal"] or not (label.get("assist_type") or "").strip():
            continue
        row = _pair(label, rows)
        if not row or not row.get("goal_confirmed", 1):
            continue
        predicted = row.get("assist_type") or "unclassified"
        pairs.append((label["assist_type"], predicted))
        correct += int(predicted == label["assist_type"])
    results["C_assist_type"] = {
        "n": len(pairs),
        "accuracy": round(correct / len(pairs), 3) if pairs else None,
        "confusion": confusion(pairs),
        "note": "no assist_type labels yet" if not pairs else "",
    }

    # --- D: the baseline we claim to beat. Needs the VSS backend, so it is opt-in.
    if with_semantic:
        try:
            from . import vss
            hits = vss.search("a cross that led to a goal", top_k=10, min_similarity=0.25)
            results["D_semantic_baseline"] = {
                "hits": len(hits.get("results", [])),
                "top_similarity": round(
                    max([h.get("similarity_score", 0) for h in hits.get("results", [{}])] or [0]), 3),
                "note": "hits are segments, not verified events: no assist_type to filter on",
            }
        except Exception as exc:  # noqa: BLE001
            results["D_semantic_baseline"] = {"error": f"{type(exc).__name__}: {exc}"}

    summary = {
        "labelled_anchors": len(labels),
        "labelled_goals": sum(1 for r in labels if r["is_goal"]),
        "events_in_table": len(rows),
        "taxonomy": config.ASSIST_TYPES,
        "results": results,
    }

    if log_weave:
        summary["weave"] = _log_to_weave(labels, rows, results)
    return summary


def _log_to_weave(labels, rows, results) -> dict:
    """Log per-anchor predictions and the aggregate metrics as a Weave evaluation."""
    state = trace.init()
    if not state.get("enabled"):
        return {"logged": False, "reason": state.get("error") or "weave unavailable"}
    try:
        from weave import EvaluationLogger

        logger = EvaluationLogger(model="fred-lasso-v1", dataset="hand-labelled-goals")
        for label in labels:
            row = _pair(label, rows)
            predicted = {
                "detected": bool(row),
                "goal_confirmed": bool(row and row.get("goal_confirmed", 1)),
                "assist_type": (row or {}).get("assist_type", ""),
                "confidence": (row or {}).get("confidence", 0.0),
                "clip_path": (row or {}).get("clip_path", ""),
            }
            prediction = logger.log_prediction(
                inputs={"match": label["match"], "t_anchor": label["t_anchor"]},
                output=predicted,
            )
            prediction.log_score("goal_correct",
                                 predicted["goal_confirmed"] == label["is_goal"])
            if label.get("assist_type"):
                prediction.log_score("assist_correct",
                                     predicted["assist_type"] == label["assist_type"])
            prediction.finish()

        flat = {}
        for stage, metrics in results.items():
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    flat[f"{stage}.{key}"] = value
        logger.log_summary(flat)
        return {"logged": True, "project": state.get("project")}
    except Exception as exc:  # noqa: BLE001
        return {"logged": False, "reason": f"{type(exc).__name__}: {exc}"}


def format_report(summary: dict) -> str:
    if "error" in summary:
        return f"eval: {summary['error']}"
    lines = [
        "",
        f"Ground truth: {summary['labelled_anchors']} labelled anchors "
        f"({summary['labelled_goals']} real goals); {summary['events_in_table']} rows in the event table",
        "",
        f"{'stage':34s} {'precision':>9s} {'recall':>7s} {'f1':>6s}   tp/fp/fn",
    ]
    for stage in ("A_anchors_only", "B_anchors_plus_cosmos_gate"):
        m = summary["results"].get(stage)
        if m:
            lines.append(f"{stage:34s} {m['precision']:>9.3f} {m['recall']:>7.3f} "
                         f"{m['f1']:>6.3f}   {m['tp']}/{m['fp']}/{m['fn']}")
    assist = summary["results"].get("C_assist_type", {})
    lines.append("")
    if assist.get("accuracy") is None:
        lines.append(f"assist type: {assist.get('note', 'not evaluated')}")
    else:
        lines.append(f"assist type: accuracy {assist['accuracy']:.3f} over n={assist['n']}")
        for truth, predictions in assist.get("confusion", {}).items():
            got = ", ".join(f"{k}x{v}" for k, v in predictions.items())
            lines.append(f"   labelled {truth:18s} -> predicted {got}")
    baseline = summary["results"].get("D_semantic_baseline")
    if baseline:
        lines.append("")
        lines.append(f"semantic baseline: {json.dumps(baseline)}")
    weave_state = summary.get("weave", {})
    lines.append("")
    lines.append(f"weave: {'logged to ' + str(weave_state.get('project')) if weave_state.get('logged') else weave_state.get('reason')}")
    return "\n".join(lines)
