import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "phase_belief").is_dir():
        PROJECT_ROOT = parent
        sys.path.insert(0, str(parent))
        break

import argparse
import csv
import json
import subprocess
from statistics import mean, median


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--split-records", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-plots-per-task", type=int, default=5)

    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--save-window-assignments", action="store_true")

    return parser.parse_args()


def run_cmd(cmd):
    print()
    print("[Run]", " ".join(str(x) for x in cmd))
    subprocess.run(cmd, check=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_csv_rows(path):
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        return list(reader)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def write_csv(path, rows):
    if not rows:
        return

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def summarize_short_segments(segments_path, thresholds=(1, 2, 3, 5)):
    rows = load_jsonl(segments_path)
    all_lengths = []
    per_demo = []

    for r in rows:
        lengths = [int(s["length"]) for s in r["segments"]]
        all_lengths.extend(lengths)

        item = {
            "file_name": r["file_name"],
            "demo_name": r["demo_name"],
            "num_segments": len(lengths),
            "num_windows": sum(lengths),
            "mean_segment_length": mean(lengths) if lengths else 0.0,
            "median_segment_length": median(lengths) if lengths else 0.0,
        }

        for th in thresholds:
            n_short = sum(1 for x in lengths if x <= th)
            item[f"short_le_{th}"] = n_short
            item[f"short_ratio_le_{th}"] = n_short / max(len(lengths), 1)

        per_demo.append(item)

    summary = {
        "num_demos": len(rows),
        "num_segments": len(all_lengths),
        "num_windows": sum(all_lengths),
        "mean_segment_length": mean(all_lengths) if all_lengths else 0.0,
        "median_segment_length": median(all_lengths) if all_lengths else 0.0,
        "min_segment_length": min(all_lengths) if all_lengths else 0,
        "max_segment_length": max(all_lengths) if all_lengths else 0,
        "mean_num_segments_per_demo": mean([x["num_segments"] for x in per_demo]) if per_demo else 0.0,
    }

    for th in thresholds:
        n_short = sum(1 for x in all_lengths if x <= th)
        summary[f"short_le_{th}"] = n_short
        summary[f"short_ratio_le_{th}"] = n_short / max(len(all_lengths), 1)

    return summary, per_demo


def suggest_phase_label(row):
    motion = safe_float(row.get("action0_motion_norm_mean"))
    rot = safe_float(row.get("action0_rot_norm_mean"))
    pos_ratio = safe_float(row.get("action0_gripper_pos_ratio"))
    neg_ratio = safe_float(row.get("action0_gripper_neg_ratio"))

    if pos_ratio >= 0.80 and motion >= 0.80:
        return "闭夹爪高速持物移动/搬运"
    if pos_ratio >= 0.80 and motion < 0.80:
        return "闭夹爪慢速调整/放置/持物精调"
    if neg_ratio >= 0.80 and motion >= 0.70:
        return "开夹爪快速移动/接近/转场"
    if neg_ratio >= 0.80 and motion < 0.55 and rot >= 0.12:
        return "开夹爪低速姿态调整/精调"
    if neg_ratio >= 0.80:
        return "开夹爪稳定操作/接触/收尾"
    return "混合/过渡 phase"


def compact_phase_profile(profile_csv):
    rows = read_csv_rows(profile_csv)
    compact = []

    for r in rows:
        item = {
            "phase": int(float(r["phase"])),
            "usage_frac": round(safe_float(r["usage_frac"]), 6),
            "confidence_mean": round(safe_float(r["confidence_mean"]), 6),
            "action_mse_mean": round(safe_float(r["action_mse_mean"]), 6),
            "motion_mean": round(safe_float(r["action0_motion_norm_mean"]), 6),
            "pos_mean": round(safe_float(r["action0_pos_norm_mean"]), 6),
            "rot_mean": round(safe_float(r["action0_rot_norm_mean"]), 6),
            "gripper_mean": round(safe_float(r["action0_gripper_mean"]), 6),
            "gripper_pos_ratio": round(safe_float(r["action0_gripper_pos_ratio"]), 6),
            "gripper_neg_ratio": round(safe_float(r["action0_gripper_neg_ratio"]), 6),
            "suggested_label": suggest_phase_label(r),
        }
        compact.append(item)

    return compact


def compact_task_usage(task_csv):
    rows = read_csv_rows(task_csv)
    compact = []

    for r in rows:
        phase_frac_cols = sorted([k for k in r.keys() if k.startswith("phase_") and k.endswith("_frac")])
        phase_fracs = {k: safe_float(r[k]) for k in phase_frac_cols}

        item = {
            "file_name": r["file_name"],
            "num_windows": int(float(r["num_windows"])),
            "dominant_phase": int(float(r["dominant_phase"])),
            "num_active_phases": int(float(r["num_active_phases"])),
        }

        for k, v in phase_fracs.items():
            item[k] = round(v, 6)

        compact.append(item)

    return compact


def make_markdown_report(path, label, eval_metrics, split_summary, short_summary, phase_profile, task_usage):
    lines = []

    lines.append(f"# Phase Analysis Summary: {label}")
    lines.append("")
    lines.append("## 1. Test metrics")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")

    for k in [
        "loss",
        "action",
        "state",
        "phase_inst_max_prob",
        "phase_inst_entropy",
        "phase_usage_entropy",
    ]:
        if k in eval_metrics:
            lines.append(f"| {k} | {eval_metrics[k]} |")

    if "phase_mean" in eval_metrics:
        lines.append(f"| soft phase_mean | {eval_metrics['phase_mean']} |")

    lines.append("")
    lines.append("## 2. Demo-level segment summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")

    for k in [
        "num_demos",
        "total_windows",
        "mean_confidence",
        "mean_entropy",
        "mean_switch_count",
        "mean_num_segments",
        "mean_avg_segment_length",
        "mean_active_phases",
    ]:
        if k in split_summary:
            lines.append(f"| {k} | {split_summary[k]} |")

    if "global_phase_usage_frac" in split_summary:
        lines.append(f"| hard phase usage | {split_summary['global_phase_usage_frac']} |")

    lines.append("")
    lines.append("## 3. Short segment summary")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---:|")

    for k, v in short_summary.items():
        if k.startswith("short_ratio") or k in [
            "num_segments",
            "mean_segment_length",
            "median_segment_length",
            "mean_num_segments_per_demo",
        ]:
            lines.append(f"| {k} | {v} |")

    lines.append("")
    lines.append("## 4. Phase action profile")
    lines.append("")
    lines.append("| phase | usage | conf | motion | pos | rot | gripper | pos_ratio | neg_ratio | suggested label |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")

    for r in phase_profile:
        lines.append(
            f"| {r['phase']} | {r['usage_frac']} | {r['confidence_mean']} | "
            f"{r['motion_mean']} | {r['pos_mean']} | {r['rot_mean']} | "
            f"{r['gripper_mean']} | {r['gripper_pos_ratio']} | {r['gripper_neg_ratio']} | "
            f"{r['suggested_label']} |"
        )

    lines.append("")
    lines.append("## 5. Task phase usage")
    lines.append("")
    lines.append("| task | dominant phase | active phases |")
    lines.append("|---|---:|---:|")

    for r in task_usage:
        lines.append(
            f"| {r['file_name']} | {r['dominant_phase']} | {r['num_active_phases']} |"
        )

    lines.append("")
    lines.append("## 6. Notes")
    lines.append("")
    lines.append("- `phase_mean` is soft usage from phase probabilities.")
    lines.append("- `hard phase usage` is computed after argmax discretization.")
    lines.append("- For modality gating, prefer using continuous `phase_probs`; use hard `phase_id` mainly for visualization.")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_out = out_dir / "eval_metrics.json"
    split_out_dir = out_dir / "phase_split"
    profile_out_dir = out_dir / "phase_action_profile"

    if not args.skip_run:
        run_cmd([
            sys.executable,
            str(PROJECT_ROOT / "tools/eval/eval_checkpoint_on_index.py"),
            "--ckpt", args.ckpt,
            "--index", args.index,
            "--out", str(eval_out),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
        ])

        run_cmd([
            sys.executable,
            str(PROJECT_ROOT / "tools/eval/eval_phase_split_demos.py"),
            "--ckpt", args.ckpt,
            "--split-records", args.split_records,
            "--split", args.split,
            "--out-dir", str(split_out_dir),
            "--seq-len", str(args.seq_len),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--max-plots-per-task", str(args.max_plots_per_task),
        ])

        profile_cmd = [
            sys.executable,
            str(PROJECT_ROOT / "tools/analysis/export_phase_action_profile.py"),
            "--ckpt", args.ckpt,
            "--index", args.index,
            "--out-dir", str(profile_out_dir),
            "--batch-size", str(args.batch_size),
            "--num-workers", str(args.num_workers),
            "--device", args.device,
        ]

        if args.save_window_assignments:
            profile_cmd.append("--save-window-assignments")

        run_cmd(profile_cmd)

    split_summary_path = split_out_dir / f"{args.split}_summary.json"
    segments_path = split_out_dir / f"{args.split}_segments.jsonl"
    profile_csv = profile_out_dir / "phase_action_profile.csv"
    task_csv = profile_out_dir / "task_phase_usage.csv"

    eval_metrics = load_json(eval_out)
    split_summary = load_json(split_summary_path)
    short_summary, short_per_demo = summarize_short_segments(segments_path)
    phase_profile = compact_phase_profile(profile_csv)
    task_usage = compact_task_usage(task_csv)

    write_json(out_dir / "short_segment_summary.json", short_summary)
    write_csv(out_dir / "short_segment_per_demo.csv", short_per_demo)
    write_csv(out_dir / "phase_profile_compact.csv", phase_profile)
    write_csv(out_dir / "task_usage_compact.csv", task_usage)

    unified = {
        "label": args.label,
        "ckpt": args.ckpt,
        "index": args.index,
        "split_records": args.split_records,
        "eval_metrics": eval_metrics,
        "split_summary": split_summary,
        "short_segment_summary": short_summary,
        "phase_profile_compact": phase_profile,
        "task_usage_compact": task_usage,
        "outputs": {
            "eval_metrics": str(eval_out),
            "phase_split_dir": str(split_out_dir),
            "phase_action_profile_dir": str(profile_out_dir),
            "summary_json": str(out_dir / "summary.json"),
            "summary_report": str(out_dir / "summary_report.md"),
        },
    }

    write_json(out_dir / "summary.json", unified)

    make_markdown_report(
        path=out_dir / "summary_report.md",
        label=args.label,
        eval_metrics=eval_metrics,
        split_summary=split_summary,
        short_summary=short_summary,
        phase_profile=phase_profile,
        task_usage=task_usage,
    )

    print()
    print("Done analysis suite.")
    print("outputs:")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "summary_report.md")
    print(" ", out_dir / "phase_profile_compact.csv")
    print(" ", out_dir / "task_usage_compact.csv")
    print()
    print(json.dumps({
        "label": args.label,
        "action": eval_metrics.get("action"),
        "phase_inst_max_prob": eval_metrics.get("phase_inst_max_prob"),
        "phase_inst_entropy": eval_metrics.get("phase_inst_entropy"),
        "phase_usage_entropy": eval_metrics.get("phase_usage_entropy"),
        "soft_phase_mean": eval_metrics.get("phase_mean"),
        "hard_phase_usage": split_summary.get("global_phase_usage_frac"),
        "mean_num_segments": split_summary.get("mean_num_segments"),
        "mean_active_phases": split_summary.get("mean_active_phases"),
        "short_ratio_le_5": short_summary.get("short_ratio_le_5"),
    }, indent=2))


if __name__ == "__main__":
    main()