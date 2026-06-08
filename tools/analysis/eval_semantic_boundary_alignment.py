import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-events", type=str, required=True)
    parser.add_argument("--pred-boundaries", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--tolerance", type=int, default=8)
    return parser.parse_args()


def read_csv(path):
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def to_int_or_none(x):
    x = str(x).strip()
    if not x:
        return None
    try:
        return int(float(x))
    except Exception:
        return None


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    human_rows = read_csv(args.human_events)
    pred_rows = read_csv(args.pred_boundaries)

    events_by_demo = {}
    preds_by_demo = {}

    for r in human_rows:
        frame = to_int_or_none(r.get("event_frame", ""))
        if frame is None:
            continue
        key = (r["file_name"], r["demo_name"])
        events_by_demo.setdefault(key, []).append({
            "event_name": r["event_name"],
            "event_frame": frame,
            "notes": r.get("notes", ""),
        })

    for r in pred_rows:
        key = (r["file_name"], r["demo_name"])
        preds_by_demo.setdefault(key, []).append({
            "boundary_frame": int(r["boundary_frame"]),
            "from_phase": int(r["from_phase"]),
            "to_phase": int(r["to_phase"]),
            "prev_len": int(r["prev_len"]),
            "next_len": int(r["next_len"]),
        })

    event_eval_rows = []
    total_events = 0
    matched_events = 0

    total_preds = 0
    matched_preds = 0

    all_keys = sorted(set(events_by_demo.keys()) | set(preds_by_demo.keys()))

    for key in all_keys:
        events = events_by_demo.get(key, [])
        preds = preds_by_demo.get(key, [])

        pred_frames = [p["boundary_frame"] for p in preds]
        event_frames = [e["event_frame"] for e in events]

        total_events += len(events)
        total_preds += len(preds)

        matched_pred_indices = set()

        for e in events:
            if pred_frames:
                dists = [abs(e["event_frame"] - pf) for pf in pred_frames]
                best_i = min(range(len(dists)), key=lambda i: dists[i])
                best_dist = dists[best_i]
                best_pred = pred_frames[best_i]
                matched = best_dist <= args.tolerance
                if matched:
                    matched_events += 1
                    matched_pred_indices.add(best_i)
            else:
                best_dist = None
                best_pred = None
                matched = False

            event_eval_rows.append({
                "file_name": key[0],
                "demo_name": key[1],
                "event_name": e["event_name"],
                "event_frame": e["event_frame"],
                "nearest_pred_boundary": best_pred,
                "distance": best_dist,
                "matched": int(matched),
                "tolerance": args.tolerance,
                "notes": e.get("notes", ""),
            })

        for i, pf in enumerate(pred_frames):
            if event_frames:
                if min(abs(pf - ef) for ef in event_frames) <= args.tolerance:
                    matched_pred_indices.add(i)

        matched_preds += len(matched_pred_indices)

    recall = matched_events / max(total_events, 1)
    precision = matched_preds / max(total_preds, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    summary = {
        "tolerance": args.tolerance,
        "total_annotated_events": total_events,
        "matched_events": matched_events,
        "boundary_recall": recall,
        "total_predicted_boundaries": total_preds,
        "matched_predicted_boundaries": matched_preds,
        "boundary_precision": precision,
        "boundary_f1": f1,
    }

    with open(out_dir / "semantic_boundary_alignment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / "semantic_boundary_alignment_events.csv", "w", newline="") as f:
        fieldnames = [
            "file_name",
            "demo_name",
            "event_name",
            "event_frame",
            "nearest_pred_boundary",
            "distance",
            "matched",
            "tolerance",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(event_eval_rows)

    print(json.dumps(summary, indent=2))
    print("Saved:", out_dir)


if __name__ == "__main__":
    main()
