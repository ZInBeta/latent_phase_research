import argparse
import json
import random
from pathlib import Path

import h5py


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/root/autodl-tmp/LIBERO/datasets/libero_goal",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_4files",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def is_success_demo(demo_group):
    if "rewards" not in demo_group or "dones" not in demo_group:
        return False

    rewards = demo_group["rewards"][:]
    dones = demo_group["dones"][:]

    if len(rewards) == 0 or len(dones) == 0:
        return False

    return float(rewards[-1]) > 0.5 and int(dones[-1]) == 1


def get_demo_length(demo_group):
    return int(demo_group["actions"].shape[0])


def build_indices(args):
    random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(data_dir.glob("*.hdf5"))

    if len(hdf5_files) == 0:
        raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

    demo_records = []
    all_windows = []

    for file_id, hdf5_path in enumerate(hdf5_files):
        print(f"[File] {hdf5_path.name}")

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"  Skip: no data group")
                continue

            demo_names = sorted(f["data"].keys())

            for demo_name in demo_names:
                demo_group = f["data"][demo_name]

                if "actions" not in demo_group:
                    continue
                if "obs" not in demo_group:
                    continue

                T = get_demo_length(demo_group)
                success = is_success_demo(demo_group)

                record = {
                    "file_id": file_id,
                    "file_path": str(hdf5_path),
                    "file_name": hdf5_path.name,
                    "demo_name": demo_name,
                    "length": T,
                    "success": success,
                }
                demo_records.append(record)

                if not success:
                    continue

                max_start = T - args.seq_len - args.pred_len

                if max_start < 0:
                    continue

                for start in range(max_start + 1):
                    window = {
                        "file_id": file_id,
                        "file_path": str(hdf5_path),
                        "file_name": hdf5_path.name,
                        "demo_name": demo_name,
                        "start": start,
                        "seq_len": args.seq_len,
                        "pred_len": args.pred_len,
                        "current_t": start + args.seq_len - 1,
                        "future_start": start + args.seq_len,
                        "future_end": start + args.seq_len + args.pred_len,
                    }
                    all_windows.append(window)

    random.shuffle(all_windows)

    n_total = len(all_windows)
    n_val = int(n_total * args.val_ratio)

    val_index = all_windows[:n_val]
    train_index = all_windows[n_val:]

    config = {
        "data_dir": str(data_dir),
        "num_hdf5_files": len(hdf5_files),
        "hdf5_files": [str(p) for p in hdf5_files],
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_demos": len(demo_records),
        "num_windows_total": n_total,
        "num_windows_train": len(train_index),
        "num_windows_val": len(val_index),
        "note": "states is not used for training input. This file only stores indices.",
    }

    with open(out_dir / "train_index.json", "w") as f:
        json.dump(train_index, f, indent=2)

    with open(out_dir / "val_index.json", "w") as f:
        json.dump(val_index, f, indent=2)

    with open(out_dir / "demo_records.json", "w") as f:
        json.dump(demo_records, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("Done.")
    print(f"Output dir: {out_dir}")
    print(f"Total demos: {len(demo_records)}")
    print(f"Total windows: {n_total}")
    print(f"Train windows: {len(train_index)}")
    print(f"Val windows: {len(val_index)}")


def main():
    args = parse_args()
    build_indices(args)


if __name__ == "__main__":
    main()