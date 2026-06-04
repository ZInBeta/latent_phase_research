import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phase_belief.models.phase_belief_bottleneck_gru import PhaseBeliefBottleneckGRU


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/checkpoints/lowdim_phase_bottleneck_k4_6files/best.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/LIBERO/datasets/libero_goal",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/outputs/batch_eval_6files",
    )
    parser.add_argument("--max-demos-per-file", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def demo_sort_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def build_windows(demo, seq_len, pred_len):
    obs = demo["obs"]

    ee_pos = obs["ee_pos"][:].astype(np.float32)
    ee_ori = obs["ee_ori"][:].astype(np.float32)
    gripper = obs["gripper_states"][:].astype(np.float32)
    actions = demo["actions"][:].astype(np.float32)

    state = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)

    T = actions.shape[0]
    max_start = T - seq_len - pred_len

    if max_start < 0:
        return None, None

    xs = []
    current_ts = []

    for start in range(max_start + 1):
        state_seq = state[start:start + seq_len]

        prev_action_seq = np.zeros((seq_len, actions.shape[-1]), dtype=np.float32)
        prev_action_seq[1:] = actions[start:start + seq_len - 1]

        x = np.concatenate([state_seq, prev_action_seq], axis=-1)
        xs.append(x)
        current_ts.append(start + seq_len - 1)

    return np.stack(xs, axis=0), np.array(current_ts)


@torch.no_grad()
def infer_demo(model, xs, device, batch_size):
    probs_all = []

    for i in range(0, xs.shape[0], batch_size):
        x = torch.from_numpy(xs[i:i + batch_size]).to(device)
        out = model(x)
        probs = out["phase_probs"][:, -1]
        probs_all.append(probs.detach().cpu())

    return torch.cat(probs_all, dim=0).numpy()


def get_segments(phase_id, current_ts, confidence):
    segments = []

    if len(phase_id) == 0:
        return segments

    start = 0
    cur_phase = int(phase_id[0])

    for i in range(1, len(phase_id)):
        if int(phase_id[i]) != cur_phase:
            end = i - 1
            segments.append({
                "phase": cur_phase,
                "start_index": int(start),
                "end_index": int(end),
                "start_frame": int(current_ts[start]),
                "end_frame": int(current_ts[end]),
                "length": int(end - start + 1),
                "mean_confidence": float(confidence[start:end + 1].mean()),
            })
            start = i
            cur_phase = int(phase_id[i])

    end = len(phase_id) - 1
    segments.append({
        "phase": cur_phase,
        "start_index": int(start),
        "end_index": int(end),
        "start_frame": int(current_ts[start]),
        "end_frame": int(current_ts[end]),
        "length": int(end - start + 1),
        "mean_confidence": float(confidence[start:end + 1].mean()),
    })

    return segments


def summarize_rows(rows, num_phases):
    total_usage = np.zeros(num_phases, dtype=np.int64)

    for row in rows:
        total_usage += np.array(row["phase_usage_counts"], dtype=np.int64)

    total_windows = int(total_usage.sum())
    usage_frac = total_usage / max(total_windows, 1)

    summary = {
        "num_demos": len(rows),
        "total_windows": total_windows,
        "global_phase_usage_counts": total_usage.tolist(),
        "global_phase_usage_frac": [round(float(x), 6) for x in usage_frac.tolist()],
        "mean_confidence": float(np.mean([r["mean_confidence"] for r in rows])),
        "mean_entropy": float(np.mean([r["mean_entropy"] for r in rows])),
        "mean_switch_count": float(np.mean([r["switch_count"] for r in rows])),
        "mean_num_segments": float(np.mean([r["num_segments"] for r in rows])),
        "mean_avg_segment_length": float(np.mean([r["avg_segment_length"] for r in rows])),
        "mean_active_phases": float(np.mean([r["active_phases"] for r in rows])),
    }

    return summary


def summarize_by_task(rows, num_phases):
    task_names = sorted(set(r["file_name"] for r in rows))
    task_summaries = []

    for task in task_names:
        task_rows = [r for r in rows if r["file_name"] == task]
        s = summarize_rows(task_rows, num_phases)
        s["file_name"] = task
        task_summaries.append(s)

    return task_summaries


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["args"]

    model = PhaseBeliefBottleneckGRU(
        input_dim=cfg["input_dim"],
        action_dim=cfg["action_dim"],
        state_dim=cfg["state_dim"],
        pred_len=cfg["pred_len"],
        num_phases=cfg["num_phases"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    num_phases = int(cfg["num_phases"])
    pred_len = int(cfg["pred_len"])

    data_dir = Path(args.data_dir)
    hdf5_files = sorted(data_dir.glob("*.hdf5"))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No hdf5 files found in {data_dir}")

    rows = []
    segment_records = []

    for hdf5_path in hdf5_files:
        print(f"[File] {hdf5_path.name}")

        with h5py.File(hdf5_path, "r") as f:
            demo_names = sorted(f["data"].keys(), key=demo_sort_key)

            if args.max_demos_per_file > 0:
                demo_names = demo_names[:args.max_demos_per_file]

            for demo_name in demo_names:
                demo = f["data"][demo_name]

                xs, current_ts = build_windows(
                    demo,
                    seq_len=args.seq_len,
                    pred_len=pred_len,
                )

                if xs is None:
                    print(f"  skip {demo_name}: too short")
                    continue

                probs = infer_demo(
                    model=model,
                    xs=xs,
                    device=device,
                    batch_size=args.batch_size,
                )

                phase_id = probs.argmax(axis=-1)
                confidence = probs.max(axis=-1)
                entropy = -(probs * np.log(probs + 1e-8)).sum(axis=-1)

                usage_counts = np.bincount(phase_id, minlength=num_phases)
                usage_frac = usage_counts / max(int(usage_counts.sum()), 1)

                switch_count = int(np.sum(phase_id[1:] != phase_id[:-1])) if len(phase_id) > 1 else 0
                num_segments = switch_count + 1 if len(phase_id) > 0 else 0
                avg_segment_length = float(len(phase_id) / max(num_segments, 1))
                active_phases = int(np.sum(usage_counts > 0))

                row = {
                    "file_name": hdf5_path.name,
                    "demo_name": demo_name,
                    "num_windows": int(len(phase_id)),
                    "mean_confidence": float(confidence.mean()),
                    "mean_entropy": float(entropy.mean()),
                    "switch_count": switch_count,
                    "num_segments": num_segments,
                    "avg_segment_length": avg_segment_length,
                    "active_phases": active_phases,
                    "phase_usage_counts": usage_counts.tolist(),
                    "phase_usage_frac": [float(x) for x in usage_frac.tolist()],
                }

                rows.append(row)

                segments = get_segments(phase_id, current_ts, confidence)
                segment_records.append({
                    "file_name": hdf5_path.name,
                    "demo_name": demo_name,
                    "segments": segments,
                })

                print(
                    f"  {demo_name}: "
                    f"conf={row['mean_confidence']:.3f} "
                    f"ent={row['mean_entropy']:.3f} "
                    f"switch={switch_count} "
                    f"active={active_phases} "
                    f"usage={usage_counts.tolist()}"
                )

    per_demo_csv = out_dir / "per_demo_metrics.csv"
    with open(per_demo_csv, "w", newline="") as f:
        fieldnames = [
            "file_name",
            "demo_name",
            "num_windows",
            "mean_confidence",
            "mean_entropy",
            "switch_count",
            "num_segments",
            "avg_segment_length",
            "active_phases",
            "phase_usage_counts",
            "phase_usage_frac",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            csv_row = row.copy()
            csv_row["phase_usage_counts"] = json.dumps(csv_row["phase_usage_counts"])
            csv_row["phase_usage_frac"] = json.dumps(csv_row["phase_usage_frac"])
            writer.writerow(csv_row)

    summary = summarize_rows(rows, num_phases)
    summary["ckpt"] = args.ckpt
    summary["data_dir"] = args.data_dir
    summary["max_demos_per_file"] = args.max_demos_per_file
    summary["seq_len"] = args.seq_len
    summary["num_phases"] = num_phases

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    task_summaries = summarize_by_task(rows, num_phases)

    task_csv = out_dir / "task_summary.csv"
    with open(task_csv, "w", newline="") as f:
        fieldnames = [
            "file_name",
            "num_demos",
            "total_windows",
            "mean_confidence",
            "mean_entropy",
            "mean_switch_count",
            "mean_num_segments",
            "mean_avg_segment_length",
            "mean_active_phases",
            "global_phase_usage_counts",
            "global_phase_usage_frac",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for s in task_summaries:
            row = s.copy()
            row["global_phase_usage_counts"] = json.dumps(row["global_phase_usage_counts"])
            row["global_phase_usage_frac"] = json.dumps(row["global_phase_usage_frac"])
            writer.writerow(row)

    with open(out_dir / "segments.jsonl", "w") as f:
        for rec in segment_records:
            f.write(json.dumps(rec) + "\n")

    print()
    print("Done.")
    print("outputs:")
    print(" ", per_demo_csv)
    print(" ", out_dir / "task_summary.csv")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "segments.jsonl")
    print()
    print("summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
