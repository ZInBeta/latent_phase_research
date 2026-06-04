import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from phase_belief.models.phase_belief_gru import PhaseBeliefGRU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="/root/autodl-tmp/phase_belief_libero/checkpoints/lowdim_phase_gru/best.pt")
    parser.add_argument("--hdf5", type=str, default="/root/autodl-tmp/LIBERO/datasets/libero_goal/put_the_bowl_on_the_plate_demo.hdf5")
    parser.add_argument("--demo", type=str, default="demo_0")
    parser.add_argument("--out-dir", type=str, default="/root/autodl-tmp/phase_belief_libero/outputs/phase_viz")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_windows(demo, seq_len, pred_len):
    obs = demo["obs"]

    ee_pos = obs["ee_pos"][:].astype(np.float32)
    ee_ori = obs["ee_ori"][:].astype(np.float32)
    gripper = obs["gripper_states"][:].astype(np.float32)
    actions = demo["actions"][:].astype(np.float32)

    state = np.concatenate([ee_pos, ee_ori, gripper], axis=-1)

    T = actions.shape[0]
    max_start = T - seq_len - pred_len

    xs = []
    current_ts = []

    for start in range(max_start + 1):
        state_seq = state[start:start + seq_len]

        prev_action_seq = np.zeros((seq_len, actions.shape[-1]), dtype=np.float32)
        prev_action_seq[1:] = actions[start:start + seq_len - 1]

        x = np.concatenate([state_seq, prev_action_seq], axis=-1)
        xs.append(x)
        current_ts.append(start + seq_len - 1)

    xs = np.stack(xs, axis=0)
    current_ts = np.array(current_ts)

    return xs, current_ts, actions, gripper


@torch.no_grad()
def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["args"]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    model = PhaseBeliefGRU(
        input_dim=cfg["input_dim"],
        action_dim=cfg["action_dim"],
        state_dim=cfg["state_dim"],
        pred_len=cfg["pred_len"],
        num_phases=cfg["num_phases"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    with h5py.File(args.hdf5, "r") as f:
        demo = f["data"][args.demo]
        xs, current_ts, actions, gripper = build_windows(
            demo,
            seq_len=16,
            pred_len=cfg["pred_len"],
        )

    x = torch.from_numpy(xs).to(device)

    probs_all = []

    batch_size = 512
    for i in range(0, x.shape[0], batch_size):
        out = model(x[i:i + batch_size])
        p_last = out["phase_probs"][:, -1]
        probs_all.append(p_last.cpu())

    probs = torch.cat(probs_all, dim=0).numpy()

    phase_id = probs.argmax(axis=-1)
    confidence = probs.max(axis=-1)
    entropy = -(probs * np.log(probs + 1e-8)).sum(axis=-1)

    stem = f"{Path(args.hdf5).stem}_{args.demo}"

    plt.figure(figsize=(14, 5))
    plt.imshow(probs.T, aspect="auto", origin="lower", interpolation="nearest")
    plt.colorbar(label="phase probability")
    plt.xlabel("window index")
    plt.ylabel("phase id")
    plt.title(f"Phase probability heatmap: {stem}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_phase_heatmap.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(current_ts, phase_id)
    plt.xlabel("frame")
    plt.ylabel("argmax phase")
    plt.title(f"Argmax phase over time: {stem}")
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_phase_argmax.png", dpi=200)
    plt.close()

    plt.figure(figsize=(14, 4))
    plt.plot(current_ts, confidence, label="max probability")
    plt.plot(current_ts, entropy, label="entropy")
    plt.xlabel("frame")
    plt.title(f"Phase confidence / entropy: {stem}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_phase_conf_entropy.png", dpi=200)
    plt.close()

    action_norm = np.linalg.norm(actions[:, :6], axis=-1)

    plt.figure(figsize=(14, 4))
    plt.plot(np.arange(len(action_norm)), action_norm, label="action norm")
    plt.plot(np.arange(len(gripper)), gripper[:, 0], label="gripper state 0")
    plt.xlabel("frame")
    plt.title(f"Action / gripper signal: {stem}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_action_gripper.png", dpi=200)
    plt.close()

    print("saved to:", out_dir)
    print("phase usage in this demo:", np.bincount(phase_id, minlength=cfg["num_phases"]))
    print("mean confidence:", float(confidence.mean()))
    print("mean entropy:", float(entropy.mean()))


if __name__ == "__main__":
    main()