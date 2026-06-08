import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


ACTION_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper_action"]


DEFAULT_DEMOS = [
    ("open_the_middle_drawer_of_the_cabinet_demo.hdf5", "demo_1"),
    ("open_the_top_drawer_and_put_the_bowl_inside_demo.hdf5", "demo_5"),
    ("push_the_plate_to_the_front_of_the_stove_demo.hdf5", "demo_13"),
    ("push_the_plate_to_the_front_of_the_stove_demo.hdf5", "demo_20"),
    ("put_the_bowl_on_the_plate_demo.hdf5", "demo_0"),
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-records", type=str, required=True)
    parser.add_argument("--segments", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--radius", type=int, default=8)
    parser.add_argument(
        "--demo",
        action="append",
        default=[],
        help="Format: file_name:demo_name. If omitted, use default demos.",
    )
    return parser.parse_args()


def load_jsonl(path):
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_boundaries(segments):
    boundaries = []
    for i in range(1, len(segments)):
        prev_s = segments[i - 1]
        cur_s = segments[i]
        boundaries.append({
            "boundary_frame": int(cur_s["start_frame"]),
            "from_phase": int(prev_s["phase"]),
            "to_phase": int(cur_s["phase"]),
            "prev_len": int(prev_s["length"]),
            "next_len": int(cur_s["length"]),
        })
    return boundaries


def norm(x):
    return float(np.linalg.norm(x))


def mean_or_nan(xs):
    if len(xs) == 0:
        return float("nan")
    return float(np.mean(xs))


def make_brief_interpretation(summary):
    delta_motion = summary["after_motion_norm_mean"] - summary["before_motion_norm_mean"]
    delta_pos = summary["after_pos_norm_mean"] - summary["before_pos_norm_mean"]
    delta_rot = summary["after_rot_norm_mean"] - summary["before_rot_norm_mean"]
    delta_grip = summary["after_gripper_abs_mean"] - summary["before_gripper_abs_mean"]

    parts = []

    if abs(delta_motion) > 0.12:
        if delta_motion > 0:
            parts.append("整体动作幅度上升：可能是开始发力、开始推动/拉动/搬运")
        else:
            parts.append("整体动作幅度下降：可能是停顿、脱离接触、接近完成或进入调整")

    if abs(delta_pos) > abs(delta_rot) and abs(delta_pos) > 0.08:
        if delta_pos > 0:
            parts.append("平移动作增强：更像移动/推动/搬运开始")
        else:
            parts.append("平移动作减弱：更像停止移动、脱离、收尾或微调")

    if abs(delta_rot) >= abs(delta_pos) and abs(delta_rot) > 0.08:
        if delta_rot > 0:
            parts.append("旋转动作增强：更像末端姿态调整/换角度")
        else:
            parts.append("旋转动作减弱：可能从姿态调整切回平移/稳定操作")

    if abs(delta_grip) > 0.05:
        if delta_grip > 0:
            parts.append("夹爪命令变化增强：可能和抓取/释放相关")
        else:
            parts.append("夹爪命令变化减弱：夹爪趋于稳定")

    if not parts:
        parts.append("前后8帧动作统计变化不大：该边界可能来自更细的时序模式或状态变化")

    return "；".join(parts)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.demo:
        target_demos = []
        for x in args.demo:
            file_name, demo_name = x.split(":", 1)
            target_demos.append((file_name, demo_name))
    else:
        target_demos = DEFAULT_DEMOS

    with open(args.split_records, "r") as f:
        split_records = json.load(f)

    file_path_map = {}
    for split_name, records in split_records.items():
        for r in records:
            key = (r["file_name"], r["demo_name"])
            file_path_map[key] = r["file_path"]

    seg_rows = load_jsonl(args.segments)
    seg_map = {(r["file_name"], r["demo_name"]): r["segments"] for r in seg_rows}

    raw_rows = []
    summary_rows = []

    for file_name, demo_name in target_demos:
        key = (file_name, demo_name)

        if key not in seg_map:
            print("missing segments:", key)
            continue

        if key not in file_path_map:
            print("missing file path:", key)
            continue

        file_path = file_path_map[key]
        segments = seg_map[key]
        boundaries = get_boundaries(segments)

        with h5py.File(file_path, "r") as f:
            actions = f["data"][demo_name]["actions"][:]

        T = len(actions)

        for b in boundaries:
            bf = int(b["boundary_frame"])
            start = max(0, bf - args.radius)
            end = min(T - 1, bf + args.radius)

            before_frames = list(range(max(0, bf - args.radius), bf))
            after_frames = list(range(bf, min(T, bf + args.radius)))

            before_actions = actions[before_frames] if before_frames else np.zeros((0, actions.shape[1]))
            after_actions = actions[after_frames] if after_frames else np.zeros((0, actions.shape[1]))

            def pos_norm_arr(a):
                return np.linalg.norm(a[:, :3], axis=1) if len(a) else np.array([])

            def rot_norm_arr(a):
                return np.linalg.norm(a[:, 3:6], axis=1) if len(a) else np.array([])

            def motion_norm_arr(a):
                return np.linalg.norm(a[:, :6], axis=1) if len(a) else np.array([])

            summary = {
                "file_name": file_name,
                "demo_name": demo_name,
                "boundary_frame": bf,
                "transition": f"{b['from_phase']}->{b['to_phase']}",
                "from_phase": b["from_phase"],
                "to_phase": b["to_phase"],
                "prev_len": b["prev_len"],
                "next_len": b["next_len"],
                "before_pos_norm_mean": mean_or_nan(pos_norm_arr(before_actions)),
                "after_pos_norm_mean": mean_or_nan(pos_norm_arr(after_actions)),
                "before_rot_norm_mean": mean_or_nan(rot_norm_arr(before_actions)),
                "after_rot_norm_mean": mean_or_nan(rot_norm_arr(after_actions)),
                "before_motion_norm_mean": mean_or_nan(motion_norm_arr(before_actions)),
                "after_motion_norm_mean": mean_or_nan(motion_norm_arr(after_actions)),
                "before_gripper_mean": mean_or_nan(before_actions[:, 6]) if len(before_actions) else float("nan"),
                "after_gripper_mean": mean_or_nan(after_actions[:, 6]) if len(after_actions) else float("nan"),
                "before_gripper_abs_mean": mean_or_nan(np.abs(before_actions[:, 6])) if len(before_actions) else float("nan"),
                "after_gripper_abs_mean": mean_or_nan(np.abs(after_actions[:, 6])) if len(after_actions) else float("nan"),
            }

            for i, name in enumerate(ACTION_NAMES):
                summary[f"before_{name}_mean"] = mean_or_nan(before_actions[:, i]) if len(before_actions) else float("nan")
                summary[f"after_{name}_mean"] = mean_or_nan(after_actions[:, i]) if len(after_actions) else float("nan")
                summary[f"before_abs_{name}_mean"] = mean_or_nan(np.abs(before_actions[:, i])) if len(before_actions) else float("nan")
                summary[f"after_abs_{name}_mean"] = mean_or_nan(np.abs(after_actions[:, i])) if len(after_actions) else float("nan")

            summary["brief_interpretation"] = make_brief_interpretation(summary)
            summary_rows.append(summary)

            for frame in range(start, end + 1):
                a = actions[frame]
                row = {
                    "file_name": file_name,
                    "demo_name": demo_name,
                    "boundary_frame": bf,
                    "transition": f"{b['from_phase']}->{b['to_phase']}",
                    "frame": frame,
                    "relative_frame": frame - bf,
                    "pos_norm": norm(a[:3]),
                    "rot_norm": norm(a[3:6]),
                    "motion_norm": norm(a[:6]),
                }
                for i, name in enumerate(ACTION_NAMES):
                    row[name] = float(a[i])
                raw_rows.append(row)

    raw_csv = out_dir / "action_around_boundaries_raw.csv"
    summary_csv = out_dir / "action_around_boundaries_summary.csv"

    if raw_rows:
        with open(raw_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(raw_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_rows)

    if summary_rows:
        with open(summary_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    print("Done.")
    print("raw:", raw_csv)
    print("summary:", summary_csv)
    print()
    for r in summary_rows:
        print(
            f"{r['file_name']} / {r['demo_name']} "
            f"boundary={r['boundary_frame']} {r['transition']} | "
            f"motion {r['before_motion_norm_mean']:.3f}->{r['after_motion_norm_mean']:.3f}, "
            f"pos {r['before_pos_norm_mean']:.3f}->{r['after_pos_norm_mean']:.3f}, "
            f"rot {r['before_rot_norm_mean']:.3f}->{r['after_rot_norm_mean']:.3f}, "
            f"grip {r['before_gripper_mean']:.3f}->{r['after_gripper_mean']:.3f} | "
            f"{r['brief_interpretation']}"
        )


if __name__ == "__main__":
    main()
