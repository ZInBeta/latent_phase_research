import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=str, required=True)
    parser.add_argument("--acb", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--thresholds", type=int, nargs="+", default=[1, 2, 3, 5])
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def summarize_one(label, rows, thresholds):
    all_lengths = []
    per_demo_rows = []

    for r in rows:
        file_name = r["file_name"]
        demo_name = r["demo_name"]
        segs = r["segments"]

        lengths = [int(s["length"]) for s in segs]
        all_lengths.extend(lengths)

        demo_row = {
            "label": label,
            "file_name": file_name,
            "demo_name": demo_name,
            "num_segments": len(lengths),
            "num_windows": sum(lengths),
            "mean_segment_length": mean(lengths) if lengths else 0.0,
            "median_segment_length": median(lengths) if lengths else 0.0,
            "min_segment_length": min(lengths) if lengths else 0,
            "max_segment_length": max(lengths) if lengths else 0,
        }

        for th in thresholds:
            n_short = sum(1 for x in lengths if x <= th)
            demo_row[f"short_le_{th}"] = n_short
            demo_row[f"short_ratio_le_{th}"] = n_short / max(len(lengths), 1)

        per_demo_rows.append(demo_row)

    summary = {
        "label": label,
        "num_demos": len(rows),
        "num_segments": len(all_lengths),
        "num_windows": sum(all_lengths),
        "mean_segment_length": mean(all_lengths) if all_lengths else 0.0,
        "median_segment_length": median(all_lengths) if all_lengths else 0.0,
        "min_segment_length": min(all_lengths) if all_lengths else 0,
        "max_segment_length": max(all_lengths) if all_lengths else 0,
        "mean_num_segments_per_demo": mean([r["num_segments"] for r in per_demo_rows]) if per_demo_rows else 0.0,
    }

    for th in thresholds:
        n_short = sum(1 for x in all_lengths if x <= th)
        summary[f"short_le_{th}"] = n_short
        summary[f"short_ratio_le_{th}_global"] = n_short / max(len(all_lengths), 1)
        summary[f"short_ratio_le_{th}_mean_per_demo"] = mean(
            [r[f"short_ratio_le_{th}"] for r in per_demo_rows]
        ) if per_demo_rows else 0.0

    return summary, per_demo_rows


def summarize_by_task(label, rows, thresholds):
    by_task = {}

    for r in rows:
        by_task.setdefault(r["file_name"], []).append(r)

    task_rows = []

    for file_name, task_items in sorted(by_task.items()):
        all_lengths = []

        for r in task_items:
            all_lengths.extend([int(s["length"]) for s in r["segments"]])

        row = {
            "label": label,
            "file_name": file_name,
            "num_demos": len(task_items),
            "num_segments": len(all_lengths),
            "num_windows": sum(all_lengths),
            "mean_segment_length": mean(all_lengths) if all_lengths else 0.0,
            "median_segment_length": median(all_lengths) if all_lengths else 0.0,
            "min_segment_length": min(all_lengths) if all_lengths else 0,
            "max_segment_length": max(all_lengths) if all_lengths else 0,
        }

        for th in thresholds:
            n_short = sum(1 for x in all_lengths if x <= th)
            row[f"short_le_{th}"] = n_short
            row[f"short_ratio_le_{th}"] = n_short / max(len(all_lengths), 1)

        task_rows.append(row)

    return task_rows


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    full_rows = load_jsonl(args.full)
    acb_rows = load_jsonl(args.acb)

    full_summary, full_per_demo = summarize_one("full", full_rows, args.thresholds)
    acb_summary, acb_per_demo = summarize_one("action_conf_balance", acb_rows, args.thresholds)

    full_task = summarize_by_task("full", full_rows, args.thresholds)
    acb_task = summarize_by_task("action_conf_balance", acb_rows, args.thresholds)

    summary_rows = [full_summary, acb_summary]

    with open(out_dir / "short_segment_summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    write_csv(out_dir / "short_segment_summary.csv", summary_rows)
    write_csv(out_dir / "short_segment_per_demo.csv", full_per_demo + acb_per_demo)
    write_csv(out_dir / "short_segment_by_task.csv", full_task + acb_task)

    print("Done.")
    print("outputs:")
    print(" ", out_dir / "short_segment_summary.json")
    print(" ", out_dir / "short_segment_summary.csv")
    print(" ", out_dir / "short_segment_per_demo.csv")
    print(" ", out_dir / "short_segment_by_task.csv")
    print()
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
