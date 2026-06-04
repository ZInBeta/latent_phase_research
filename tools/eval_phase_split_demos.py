import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phase_belief.models.phase_belief_bottleneck_gru import PhaseBeliefBottleneckGRU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--split-records", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-plots-per-task", type=int, default=3)
    return parser.parse_args()


def demo_sort_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def safe_stem(file_name, demo_name):
    return f"{Path(file_name).stem}_{demo_name}"


def build_windows_and_signals(demo, seq_len, pred_len):
    obs = demo["obs"]

    ee_pos = obs["ee_pos"][:].astype(np.float32)
    ee_ori = obs["ee_ori"][:].astype(np.float32)
    gripper = obs["gripper_states"][:].astype(np.float32)
    actions = demo["actions"][:].astype(np.float32)

    state = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)

    T = actions.shape[0]
    max_start = T - seq_len - pred_len

    if max_start < 0:
        return None

    xs = []
    current_ts = []

    for start in range(max_start + 1):
        state_seq = state[start:start + seq_len]

        prev_action_seq = np.zeros((seq_len, actions.shape[-1]), dtype=np.float32)
        prev_action_seq[1:] = actions[start:start + seq_len - 1]

        x = np.concatenate([state_seq, prev_action_seq], axis=-1)
        xs.append(x)
        current_ts.append(start + seq_len - 1)

    return {
        "xs": np.stack(xs, axis=0),
        "current_ts": np.array(current_ts),
        "actions": actions,
        "gripper": gripper,
    }


@torch.no_grad()
def infer_demo(model, xs, device, batch_size):
    probs_all = []

    for i in range(0, xs.shape[0], batch_size):
        x = torch.from_numpy(xs[i:i + batch_size]).to(device)
        out = model(x)
        probs_all.append(out["phase_probs"][:, -1].detach().cpu())

    return torch.cat(probs_all, dim=0).numpy()


def get_segments(phase_id, current_ts, confidence):
    if len(phase_id) == 0:
        return []

    segments = []
    start = 0
    cur = int(phase_id[0])

    for i in range(1, len(phase_id)):
        if int(phase_id[i]) != cur:
            end = i - 1
            segments.append({
                "phase": cur,
                "start_index": int(start),
                "end_index": int(end),
                "start_frame": int(current_ts[start]),
                "end_frame": int(current_ts[end]),
                "length": int(end - start + 1),
                "mean_confidence": float(confidence[start:end + 1].mean()),
            })
            start = i
            cur = int(phase_id[i])

    end = len(phase_id) - 1
    segments.append({
        "phase": cur,
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

    for r in rows:
        total_usage += np.array(r["phase_usage_counts"], dtype=np.int64)

    total_windows = int(total_usage.sum())
    usage_frac = total_usage / max(total_windows, 1)

    return {
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


def summarize_by_task(rows, num_phases):
    out = []
    for file_name in sorted(set(r["file_name"] for r in rows)):
        task_rows = [r for r in rows if r["file_name"] == file_name]
        s = summarize_rows(task_rows, num_phases)
        s["file_name"] = file_name
        out.append(s)
    return out


# def plot_demo(out_dir, file_name, demo_name, probs, current_ts, actions, gripper):
#     stem = safe_stem(file_name, demo_name)

#     phase_id = probs.argmax(axis=-1)
#     confidence = probs.max(axis=-1)
#     entropy = -(probs * np.log(probs + 1e-8)).sum(axis=-1)

#     plt.figure(figsize=(14, 5))
#     plt.imshow(probs.T, aspect="auto", origin="lower", interpolation="nearest")
#     plt.colorbar(label="phase probability")
#     plt.xlabel("window index")
#     plt.ylabel("phase id")
#     plt.title(f"Phase probability heatmap: {stem}")
#     plt.tight_layout()
#     plt.savefig(out_dir / f"{stem}_phase_heatmap.png", dpi=200)
#     plt.close()

#     plt.figure(figsize=(14, 4))
#     plt.plot(current_ts, phase_id)
#     plt.xlabel("frame")
#     plt.ylabel("argmax phase")
#     plt.title(f"Argmax phase over time: {stem}")
#     plt.tight_layout()
#     plt.savefig(out_dir / f"{stem}_phase_argmax.png", dpi=200)
#     plt.close()

#     plt.figure(figsize=(14, 4))
#     plt.plot(current_ts, confidence, label="max probability")
#     plt.plot(current_ts, entropy, label="entropy")
#     plt.xlabel("frame")
#     plt.title(f"Phase confidence / entropy: {stem}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_dir / f"{stem}_phase_conf_entropy.png", dpi=200)
#     plt.close()

#     action_norm = np.linalg.norm(actions[:, :6], axis=-1)

#     plt.figure(figsize=(14, 4))
#     plt.plot(np.arange(len(action_norm)), action_norm, label="action norm")
#     plt.plot(np.arange(len(gripper)), gripper[:, 0], label="gripper state 0")
#     plt.xlabel("frame")
#     plt.title(f"Action / gripper signal: {stem}")
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(out_dir / f"{stem}_action_gripper.png", dpi=200)
#     plt.close()

def plot_demo(out_dir, file_name, demo_name, probs, current_ts, actions, gripper):
    stem = safe_stem(file_name, demo_name)

    phase_id = probs.argmax(axis=-1)
    confidence = probs.max(axis=-1)
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=-1)
    action_norm = np.linalg.norm(actions[:, :6], axis=-1)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16), constrained_layout=True)

    im = axes[0].imshow(
        probs.T,
        aspect="auto",
        origin="lower",
        interpolation="nearest"
    )
    fig.colorbar(im, ax=axes[0], label="phase probability")
    axes[0].set_xlabel("window index")
    axes[0].set_ylabel("phase id")
    axes[0].set_title(f"Phase probability heatmap: {stem}")

    axes[1].plot(current_ts, phase_id)
    axes[1].set_xlabel("frame")
    axes[1].set_ylabel("argmax phase")
    axes[1].set_title("Argmax phase over time")

    axes[2].plot(current_ts, confidence, label="max probability")
    axes[2].plot(current_ts, entropy, label="entropy")
    axes[2].set_xlabel("frame")
    axes[2].set_title("Phase confidence / entropy")
    axes[2].legend()

    axes[3].plot(np.arange(len(action_norm)), action_norm, label="action norm")
    axes[3].plot(np.arange(len(gripper)), gripper[:, 0], label="gripper state 0")
    axes[3].set_xlabel("frame")
    axes[3].set_title("Action / gripper signal")
    axes[3].legend()

    fig.savefig(out_dir / f"{stem}_combined.png", dpi=200)
    plt.close(fig)


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    plot_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

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

    with open(args.split_records, "r") as f:
        split_records = json.load(f)

    records = split_records[args.split]
    records = sorted(records, key=lambda r: (r["file_name"], demo_sort_key(r["demo_name"])))

    rows = []
    segment_records = []
    plot_count_by_task = {}

    for r in records:
        file_path = r["file_path"]
        file_name = r["file_name"]
        demo_name = r["demo_name"]

        with h5py.File(file_path, "r") as f:
            demo = f["data"][demo_name]
            pack = build_windows_and_signals(
                demo=demo,
                seq_len=args.seq_len,
                pred_len=pred_len,
            )

        if pack is None:
            print(f"skip {file_name}/{demo_name}: too short")
            continue

        probs = infer_demo(
            model=model,
            xs=pack["xs"],
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
            "file_name": file_name,
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

        segments = get_segments(phase_id, pack["current_ts"], confidence)
        segment_records.append({
            "file_name": file_name,
            "demo_name": demo_name,
            "segments": segments,
        })

        print(
            f"{file_name}/{demo_name}: "
            f"conf={row['mean_confidence']:.3f} "
            f"ent={row['mean_entropy']:.3f} "
            f"switch={switch_count} "
            f"active={active_phases} "
            f"usage={usage_counts.tolist()}"
        )

        plot_count_by_task.setdefault(file_name, 0)
        if args.max_plots_per_task < 0 or plot_count_by_task[file_name] < args.max_plots_per_task:
            plot_task_dir = plot_dir / Path(file_name).stem
            plot_task_dir.mkdir(parents=True, exist_ok=True)

            plot_demo(
                out_dir=plot_task_dir,
                file_name=file_name,
                demo_name=demo_name,
                probs=probs,
                current_ts=pack["current_ts"],
                actions=pack["actions"],
                gripper=pack["gripper"],
            )
            plot_count_by_task[file_name] += 1

    per_demo_csv = out_dir / f"{args.split}_per_demo_metrics.csv"
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

    task_summaries = summarize_by_task(rows, num_phases)
    task_csv = out_dir / f"{args.split}_task_summary.csv"

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

    summary = summarize_rows(rows, num_phases)
    summary["split"] = args.split
    summary["ckpt"] = args.ckpt
    summary["split_records"] = args.split_records
    summary["seq_len"] = args.seq_len
    summary["num_phases"] = num_phases
    summary["max_plots_per_task"] = args.max_plots_per_task

    with open(out_dir / f"{args.split}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open(out_dir / f"{args.split}_segments.jsonl", "w") as f:
        for rec in segment_records:
            f.write(json.dumps(rec) + "\n")

    print()
    print("Done.")
    print("outputs:")
    print(" ", per_demo_csv)
    print(" ", task_csv)
    print(" ", out_dir / f"{args.split}_summary.json")
    print(" ", out_dir / f"{args.split}_segments.jsonl")
    print(" ", plot_dir)
    print()
    print("summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
