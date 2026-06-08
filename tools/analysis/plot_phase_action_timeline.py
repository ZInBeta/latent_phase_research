import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignments", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--file-name", type=str, default=None)
    parser.add_argument("--demo-name", type=str, default=None)
    parser.add_argument("--demo-list", type=str, default=None)
    return parser.parse_args()


def load_demo_list(path):
    demos = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if "," in line:
                file_name, demo_name = line.split(",", 1)
            else:
                parts = line.split()
                file_name, demo_name = parts[0], parts[1]

            demos.append((file_name.strip(), demo_name.strip()))

    return demos


def sanitize_name(x):
    return (
        x.replace("/", "_")
        .replace("\\", "_")
        .replace(" ", "_")
        .replace(".hdf5", "")
    )


def plot_one_demo(df, file_name, demo_name, out_dir):
    d = df[(df["file_name"] == file_name) & (df["demo_name"] == demo_name)].copy()

    if len(d) == 0:
        print(f"[Skip] no rows: {file_name}/{demo_name}")
        return

    d = d.sort_values("current_t")

    t = d["current_t"].to_numpy()
    phase = d["phase"].to_numpy()
    conf = d["confidence"].to_numpy()

    motion = d["action0_motion_norm"].to_numpy()
    pos = d["action0_pos_norm"].to_numpy()
    rot = d["action0_rot_norm"].to_numpy()
    grip = d["action0_gripper"].to_numpy()

    phase_change_idx = d["phase"].ne(d["phase"].shift()).to_numpy()
    boundary_t = t[phase_change_idx][1:]

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

    axes[0].step(t, phase, where="post")
    axes[0].set_ylabel("phase")
    axes[0].set_yticks(sorted(d["phase"].unique()))

    axes[1].plot(t, conf)
    axes[1].set_ylabel("confidence")
    axes[1].set_ylim(0.0, 1.05)

    axes[2].plot(t, motion, label="motion")
    axes[2].plot(t, pos, label="pos")
    axes[2].plot(t, rot, label="rot")
    axes[2].set_ylabel("norm")
    axes[2].legend(loc="upper right")

    axes[3].step(t, grip, where="post")
    axes[3].set_ylabel("gripper")

    axes[4].plot(t, motion)
    axes[4].scatter(t, motion, c=phase, s=12)
    axes[4].set_ylabel("motion by phase")
    axes[4].set_xlabel("frame / current_t")

    for ax in axes:
        for bt in boundary_t:
            ax.axvline(bt, linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(True, alpha=0.3)

    title = f"{file_name} / {demo_name}"
    fig.suptitle(title)
    fig.tight_layout()

    out_name = f"{sanitize_name(file_name)}_{sanitize_name(demo_name)}_phase_action_timeline.png"
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=180)
    plt.close(fig)

    print(f"[Saved] {out_path}")


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.assignments)

    if args.demo_list is not None:
        demos = load_demo_list(args.demo_list)
    else:
        if args.file_name is None or args.demo_name is None:
            raise ValueError("Need either --demo-list or both --file-name and --demo-name")
        demos = [(args.file_name, args.demo_name)]

    for file_name, demo_name in demos:
        plot_one_demo(df, file_name, demo_name, out_dir)


if __name__ == "__main__":
    main()