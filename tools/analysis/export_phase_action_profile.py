import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "phase_belief").is_dir():
        sys.path.insert(0, str(parent))
        break

import argparse
import csv
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase_belief.data.dataset import PhaseBeliefDataset
from phase_belief.models.phase_belief_bottleneck_gru import PhaseBeliefBottleneckGRU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-window-assignments", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_mean(xs):
    return float(np.mean(xs)) if len(xs) > 0 else 0.0


def safe_std(xs):
    return float(np.std(xs)) if len(xs) > 0 else 0.0


def safe_frac(num, den):
    return float(num) / float(den) if den > 0 else 0.0


def get_phase_probs_last(phase_probs):
    if phase_probs.dim() == 3:
        return phase_probs[:, -1, :]
    if phase_probs.dim() == 2:
        return phase_probs
    raise ValueError(f"Unexpected phase_probs shape: {tuple(phase_probs.shape)}")


def build_model_from_ckpt(ckpt, device):
    args = ckpt.get("args", {})

    model = PhaseBeliefBottleneckGRU(
        input_dim=int(args.get("input_dim", 15)),
        action_dim=int(args.get("action_dim", 7)),
        state_dim=int(args.get("state_dim", 8)),
        pred_len=int(args.get("pred_len", 4)),
        num_phases=int(args.get("num_phases", 4)),
        hidden_dim=int(args.get("hidden_dim", 128)),
    )

    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    return model, args


def action_features(actions):
    """
    actions: numpy array [B, H, 7]
    action dims:
      0:3 position delta
      3:6 rotation delta
      6 gripper action
    """
    a0 = actions[:, 0, :]
    motion0 = np.linalg.norm(a0[:, :6], axis=1)
    pos0 = np.linalg.norm(a0[:, :3], axis=1)
    rot0 = np.linalg.norm(a0[:, 3:6], axis=1)
    grip0 = a0[:, 6]

    motion_h = np.linalg.norm(actions[:, :, :6], axis=2).mean(axis=1)
    pos_h = np.linalg.norm(actions[:, :, :3], axis=2).mean(axis=1)
    rot_h = np.linalg.norm(actions[:, :, 3:6], axis=2).mean(axis=1)
    grip_h = actions[:, :, 6].mean(axis=1)

    return {
        "action0_motion_norm": motion0,
        "action0_pos_norm": pos0,
        "action0_rot_norm": rot0,
        "action0_gripper": grip0,
        "future_motion_norm_mean": motion_h,
        "future_pos_norm_mean": pos_h,
        "future_rot_norm_mean": rot_h,
        "future_gripper_mean": grip_h,
    }


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print("device:", device)

    index_rows = load_json(args.index)
    demo_len_by_key = {}

    for r in index_rows:
        key = (r["file_name"], r["demo_name"])
        future_end = int(r["future_end"])
        demo_len_by_key[key] = max(demo_len_by_key.get(key, 0), future_end)

    ckpt = torch.load(args.ckpt, map_location=device)
    model, ckpt_args = build_model_from_ckpt(ckpt, device)
    num_phases = int(ckpt_args.get("num_phases", 4))

    dataset = PhaseBeliefDataset(args.index)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    phase_stats = {
        p: defaultdict(list)
        for p in range(num_phases)
    }

    task_phase_counts = defaultdict(lambda: np.zeros(num_phases, dtype=np.int64))
    task_total_counts = defaultdict(int)

    window_rows = []
    cursor = 0

    total_action_mse = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            target_actions = batch["future_actions"].to(device, non_blocking=True)

            out = model(x)

            phase_probs = get_phase_probs_last(out["phase_probs"])
            phase_conf, phase_id = phase_probs.max(dim=-1)

            pred_actions = out["pred_actions"]
            action_mse_per_sample = F.mse_loss(
                pred_actions,
                target_actions,
                reduction="none",
            ).mean(dim=(1, 2))

            target_np = target_actions.detach().cpu().numpy()
            feat = action_features(target_np)

            phase_id_np = phase_id.detach().cpu().numpy()
            phase_conf_np = phase_conf.detach().cpu().numpy()
            phase_probs_np = phase_probs.detach().cpu().numpy()
            action_mse_np = action_mse_per_sample.detach().cpu().numpy()

            bs = x.shape[0]
            meta_rows = index_rows[cursor:cursor + bs]
            cursor += bs

            total_action_mse += float(action_mse_per_sample.sum().item())
            total_samples += bs

            for i in range(bs):
                p = int(phase_id_np[i])
                meta = meta_rows[i]
                # file_name = meta["file_name"]
                # demo_name = meta["demo_name"]

                # current_t = int(meta.get("current_t", meta["start"] + meta["seq_len"] - 1))
                # demo_len = int(meta.get("length", -1))
                # if demo_len <= 1:
                #     rel_t = -1.0
                # else:
                #     rel_t = current_t / max(demo_len - 1, 1)
                file_name = meta["file_name"]
                demo_name = meta["demo_name"]
                
                current_t = int(meta.get("current_t", meta["start"] + meta["seq_len"] - 1))
                
                key = (file_name, demo_name)
                demo_len = int(meta.get("length", demo_len_by_key.get(key, -1)))
                
                if demo_len <= 1:
                    rel_t = -1.0
                else:
                    rel_t = current_t / max(demo_len - 1, 1)

                task_phase_counts[file_name][p] += 1
                task_total_counts[file_name] += 1

                phase_stats[p]["confidence"].append(float(phase_conf_np[i]))
                phase_stats[p]["action_mse"].append(float(action_mse_np[i]))
                phase_stats[p]["rel_t"].append(float(rel_t))

                for key, values in feat.items():
                    phase_stats[p][key].append(float(values[i]))

                grip0 = float(feat["action0_gripper"][i])
                grip_h = float(feat["future_gripper_mean"][i])

                phase_stats[p]["action0_gripper_pos"].append(1.0 if grip0 > 0 else 0.0)
                phase_stats[p]["action0_gripper_neg"].append(1.0 if grip0 < 0 else 0.0)
                phase_stats[p]["action0_gripper_zero"].append(1.0 if grip0 == 0 else 0.0)

                phase_stats[p]["future_gripper_pos"].append(1.0 if grip_h > 0 else 0.0)
                phase_stats[p]["future_gripper_neg"].append(1.0 if grip_h < 0 else 0.0)
                phase_stats[p]["future_gripper_zero"].append(1.0 if grip_h == 0 else 0.0)

                if args.save_window_assignments:
                    row = {
                        "file_name": file_name,
                        "demo_name": demo_name,
                        "start": meta["start"],
                        "current_t": current_t,
                        "phase": p,
                        "confidence": float(phase_conf_np[i]),
                        "rel_t": rel_t,
                        "action_mse": float(action_mse_np[i]),
                        "action0_motion_norm": float(feat["action0_motion_norm"][i]),
                        "action0_pos_norm": float(feat["action0_pos_norm"][i]),
                        "action0_rot_norm": float(feat["action0_rot_norm"][i]),
                        "action0_gripper": grip0,
                        "future_motion_norm_mean": float(feat["future_motion_norm_mean"][i]),
                        "future_pos_norm_mean": float(feat["future_pos_norm_mean"][i]),
                        "future_rot_norm_mean": float(feat["future_rot_norm_mean"][i]),
                        "future_gripper_mean": grip_h,
                    }

                    for pp in range(num_phases):
                        row[f"phase_prob_{pp}"] = float(phase_probs_np[i, pp])

                    window_rows.append(row)

    dataset.close()

    total_windows = sum(len(phase_stats[p]["confidence"]) for p in range(num_phases))

    phase_profile_rows = []
    phase_time_rows = []

    for p in range(num_phases):
        s = phase_stats[p]
        n = len(s["confidence"])

        row = {
            "phase": p,
            "num_windows": n,
            "usage_frac": safe_frac(n, total_windows),
            "confidence_mean": safe_mean(s["confidence"]),
            "confidence_std": safe_std(s["confidence"]),
            "action_mse_mean": safe_mean(s["action_mse"]),
            "action_mse_std": safe_std(s["action_mse"]),

            "action0_motion_norm_mean": safe_mean(s["action0_motion_norm"]),
            "action0_motion_norm_std": safe_std(s["action0_motion_norm"]),
            "action0_pos_norm_mean": safe_mean(s["action0_pos_norm"]),
            "action0_pos_norm_std": safe_std(s["action0_pos_norm"]),
            "action0_rot_norm_mean": safe_mean(s["action0_rot_norm"]),
            "action0_rot_norm_std": safe_std(s["action0_rot_norm"]),
            "action0_gripper_mean": safe_mean(s["action0_gripper"]),
            "action0_gripper_std": safe_std(s["action0_gripper"]),
            "action0_gripper_pos_ratio": safe_mean(s["action0_gripper_pos"]),
            "action0_gripper_neg_ratio": safe_mean(s["action0_gripper_neg"]),
            "action0_gripper_zero_ratio": safe_mean(s["action0_gripper_zero"]),

            "future_motion_norm_mean": safe_mean(s["future_motion_norm_mean"]),
            "future_motion_norm_std": safe_std(s["future_motion_norm_mean"]),
            "future_pos_norm_mean": safe_mean(s["future_pos_norm_mean"]),
            "future_pos_norm_std": safe_std(s["future_pos_norm_mean"]),
            "future_rot_norm_mean": safe_mean(s["future_rot_norm_mean"]),
            "future_rot_norm_std": safe_std(s["future_rot_norm_mean"]),
            "future_gripper_mean": safe_mean(s["future_gripper_mean"]),
            "future_gripper_std": safe_std(s["future_gripper_mean"]),
            "future_gripper_pos_ratio": safe_mean(s["future_gripper_pos"]),
            "future_gripper_neg_ratio": safe_mean(s["future_gripper_neg"]),
            "future_gripper_zero_ratio": safe_mean(s["future_gripper_zero"]),
        }
        phase_profile_rows.append(row)

        rel_ts = [x for x in s["rel_t"] if x >= 0]
        phase_time_rows.append({
            "phase": p,
            "num_windows": n,
            "rel_t_mean": safe_mean(rel_ts),
            "rel_t_std": safe_std(rel_ts),
            "rel_t_min": float(np.min(rel_ts)) if rel_ts else 0.0,
            "rel_t_max": float(np.max(rel_ts)) if rel_ts else 0.0,
            "early_ratio_rel_t_lt_0.33": safe_mean([1.0 if x < 0.33 else 0.0 for x in rel_ts]),
            "middle_ratio_0.33_to_0.66": safe_mean([1.0 if 0.33 <= x < 0.66 else 0.0 for x in rel_ts]),
            "late_ratio_rel_t_ge_0.66": safe_mean([1.0 if x >= 0.66 else 0.0 for x in rel_ts]),
        })

    task_rows = []
    for file_name in sorted(task_phase_counts.keys()):
        counts = task_phase_counts[file_name]
        total = int(task_total_counts[file_name])

        row = {
            "file_name": file_name,
            "num_windows": total,
        }

        for p in range(num_phases):
            row[f"phase_{p}_count"] = int(counts[p])
            row[f"phase_{p}_frac"] = safe_frac(int(counts[p]), total)

        row["dominant_phase"] = int(np.argmax(counts)) if total > 0 else -1
        row["num_active_phases"] = int((counts > 0).sum())

        task_rows.append(row)

    summary = {
        "ckpt": args.ckpt,
        "index": args.index,
        "num_windows": total_windows,
        "num_phases": num_phases,
        "mean_action_mse": safe_frac(total_action_mse, total_samples),
        "outputs": {
            "phase_action_profile": str(out_dir / "phase_action_profile.csv"),
            "phase_time_profile": str(out_dir / "phase_time_profile.csv"),
            "task_phase_usage": str(out_dir / "task_phase_usage.csv"),
            "window_phase_assignments": str(out_dir / "window_phase_assignments.csv") if args.save_window_assignments else None,
        },
    }

    write_csv(out_dir / "phase_action_profile.csv", phase_profile_rows)
    write_csv(out_dir / "phase_time_profile.csv", phase_time_rows)
    write_csv(out_dir / "task_phase_usage.csv", task_rows)

    if args.save_window_assignments:
        write_csv(out_dir / "window_phase_assignments.csv", window_rows)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(json.dumps(summary, indent=2))
    print()
    print("phase_action_profile:")
    print(json.dumps(phase_profile_rows, indent=2))


if __name__ == "__main__":
    main()