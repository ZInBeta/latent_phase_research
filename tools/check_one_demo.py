import argparse
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt


def save_image_strip(images, out_path, title):
    t = images.shape[0]
    idxs = np.linspace(0, t - 1, 8).astype(int)

    fig, axes = plt.subplots(1, len(idxs), figsize=(16, 2.5))
    for ax, idx in zip(axes, idxs):
        ax.imshow(images[idx])
        ax.set_title(str(idx))
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_series(arr, out_path, title, max_dims=None):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[:, None]
    if max_dims is not None:
        arr = arr[:, :max_dims]

    plt.figure(figsize=(12, 4))
    for i in range(arr.shape[1]):
        plt.plot(arr[:, i], label=f"dim_{i}")
    plt.title(title)
    plt.xlabel("time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--out", default="/root/autodl-tmp/phase_belief_libero/outputs/check")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with h5py.File(args.path, "r") as f:
        g = f[f"data/{args.demo}"]

        actions = g["actions"][:]
        dones = g["dones"][:]
        rewards = g["rewards"][:]

        obs = g["obs"]
        agentview = obs["agentview_rgb"][:]
        wrist = obs["eye_in_hand_rgb"][:]
        ee_pos = obs["ee_pos"][:]
        ee_ori = obs["ee_ori"][:]
        ee_states = obs["ee_states"][:]
        gripper = obs["gripper_states"][:]
        joint = obs["joint_states"][:]
        robot_states = g["robot_states"][:]
        states = g["states"][:]

    print("demo:", args.demo)
    print("T:", actions.shape[0])
    print("actions:", actions.shape, actions.dtype, "min", actions.min(), "max", actions.max())
    print("agentview:", agentview.shape, agentview.dtype)
    print("wrist:", wrist.shape, wrist.dtype)
    print("ee_pos:", ee_pos.shape, "min", ee_pos.min(axis=0), "max", ee_pos.max(axis=0))
    print("ee_ori:", ee_ori.shape, "min", ee_ori.min(axis=0), "max", ee_ori.max(axis=0))
    print("ee_states:", ee_states.shape)
    print("gripper:", gripper.shape, "min", gripper.min(axis=0), "max", gripper.max(axis=0))
    print("joint:", joint.shape)
    print("robot_states:", robot_states.shape)
    print("states:", states.shape)
    print("rewards sum:", rewards.sum(), "last reward:", rewards[-1])
    print("dones sum:", dones.sum(), "last done:", dones[-1])
    print("first 10 gripper:")
    print(gripper[:10])
    print("last 10 gripper:")
    print(gripper[-10:])

    prefix = os.path.join(args.out, args.demo)

    save_image_strip(agentview, prefix + "_agentview_strip.png", f"{args.demo} agentview")
    save_image_strip(wrist, prefix + "_wrist_strip.png", f"{args.demo} wrist")

    plot_series(actions, prefix + "_actions.png", f"{args.demo} actions")
    plot_series(ee_pos, prefix + "_ee_pos.png", f"{args.demo} ee_pos")
    plot_series(ee_ori, prefix + "_ee_ori.png", f"{args.demo} ee_ori")
    plot_series(gripper, prefix + "_gripper.png", f"{args.demo} gripper")
    plot_series(rewards, prefix + "_rewards.png", f"{args.demo} rewards")
    plot_series(dones, prefix + "_dones.png", f"{args.demo} dones")

    print("saved to:", args.out)


if __name__ == "__main__":
    main()
