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
        default="/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_6files_demo_split",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-num-files", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def demo_sort_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def is_success_demo(demo_group):
    if "rewards" not in demo_group or "dones" not in demo_group:
        return False

    rewards = demo_group["rewards"][:]
    dones = demo_group["dones"][:]

    if len(rewards) == 0 or len(dones) == 0:
        return False

    return float(rewards[-1]) > 0.5 and int(dones[-1]) == 1


def make_windows(record, seq_len, pred_len):
    T = int(record["length"])
    max_start = T - seq_len - pred_len

    windows = []
    if max_start < 0:
        return windows

    for start in range(max_start + 1):
        windows.append({
            "file_id": record["file_id"],
            "file_path": record["file_path"],
            "file_name": record["file_name"],
            "demo_name": record["demo_name"],
            "start": start,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "current_t": start + seq_len - 1,
            "future_start": start + seq_len,
            "future_end": start + seq_len + pred_len,
        })

    return windows


def split_records_per_file(records, train_ratio, val_ratio, seed):
    rng = random.Random(seed)

    by_file = {}
    for r in records:
        by_file.setdefault(r["file_name"], []).append(r)

    split_records = {
        "train": [],
        "val": [],
        "test": [],
    }

    for file_name, file_records in sorted(by_file.items()):
        file_records = sorted(file_records, key=lambda x: demo_sort_key(x["demo_name"]))
        rng.shuffle(file_records)

        n = len(file_records)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_records = file_records[:n_train]
        val_records = file_records[n_train:n_train + n_val]
        test_records = file_records[n_train + n_val:]

        split_records["train"].extend(train_records)
        split_records["val"].extend(val_records)
        split_records["test"].extend(test_records)

        print(
            f"[Split] {file_name}: "
            f"total={n}, train={len(train_records)}, "
            f"val={len(val_records)}, test={len(test_records)}"
        )

    return split_records


def assert_no_demo_overlap(split_records):
    sets = {}

    for split, records in split_records.items():
        sets[split] = set((r["file_name"], r["demo_name"]) for r in records)

    assert sets["train"].isdisjoint(sets["val"]), "train/val demo overlap"
    assert sets["train"].isdisjoint(sets["test"]), "train/test demo overlap"
    assert sets["val"].isdisjoint(sets["test"]), "val/test demo overlap"


def main():
    args = parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    data_dir = Path(args.data_dir)
    # out_dir = Path(args.out_dir)
    # out_dir.mkdir(parents=True, exist_ok=True)

    # hdf5_files = sorted(data_dir.glob("*.hdf5"))
    out_dir = Path(args.out_dir)

    if out_dir.exists() and not args.overwrite:
        existing_files = [
            "train_index.json",
            "val_index.json",
            "test_index.json",
            "config.json",
        ]
        if any((out_dir / name).exists() for name in existing_files):
            raise FileExistsError(
                f"Output dir already has split files: {out_dir}. "
                f"Use --overwrite if you really want to regenerate it."
            )
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if len(hdf5_files) != args.expected_num_files:
        raise ValueError(
            f"Expected {args.expected_num_files} hdf5 files, "
            f"but found {len(hdf5_files)} in {data_dir}"
        )
    # if len(hdf5_files) == 0:
    #     raise FileNotFoundError(f"No .hdf5 files found in {data_dir}")

    demo_records = []

    for file_id, hdf5_path in enumerate(hdf5_files):
        print(f"[File] {hdf5_path.name}")

        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print("  skip: no data group")
                continue

            demo_names = sorted(f["data"].keys(), key=demo_sort_key)

            for demo_name in demo_names:
                demo = f["data"][demo_name]

                if "actions" not in demo or "obs" not in demo:
                    continue

                T = int(demo["actions"].shape[0])
                success = is_success_demo(demo)
                max_start = T - args.seq_len - args.pred_len
                usable = bool(success and max_start >= 0)

                record = {
                    "file_id": file_id,
                    "file_path": str(hdf5_path),
                    "file_name": hdf5_path.name,
                    "demo_name": demo_name,
                    "length": T,
                    "success": success,
                    "usable": usable,
                }

                demo_records.append(record)

    usable_records = [r for r in demo_records if r["usable"]]

    split_records = split_records_per_file(
        usable_records,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    assert_no_demo_overlap(split_records)

    train_index = []
    val_index = []
    test_index = []

    for r in split_records["train"]:
        train_index.extend(make_windows(r, args.seq_len, args.pred_len))

    for r in split_records["val"]:
        val_index.extend(make_windows(r, args.seq_len, args.pred_len))

    for r in split_records["test"]:
        test_index.extend(make_windows(r, args.seq_len, args.pred_len))

    rng = random.Random(args.seed)
    rng.shuffle(train_index)
    rng.shuffle(val_index)
    rng.shuffle(test_index)

    split_demo_records = {
        "train": split_records["train"],
        "val": split_records["val"],
        "test": split_records["test"],
    }

    config = {
        "data_dir": str(data_dir),
        "hdf5_files": [str(p) for p in hdf5_files],
        "num_hdf5_files": len(hdf5_files),
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "test_ratio": 1.0 - args.train_ratio - args.val_ratio,
        "seed": args.seed,
        "num_demos_total": len(demo_records),
        "num_demos_usable": len(usable_records),
        "num_demos_train": len(split_records["train"]),
        "num_demos_val": len(split_records["val"]),
        "num_demos_test": len(split_records["test"]),
        "num_windows_train": len(train_index),
        "num_windows_val": len(val_index),
        "num_windows_test": len(test_index),
        "split_rule": "split demos independently within each hdf5 task file; no demo overlap across train/val/test",
        "note": "states is not used for training input. This file only stores indices.",
    }

    with open(out_dir / "train_index.json", "w") as f:
        json.dump(train_index, f, indent=2)

    with open(out_dir / "val_index.json", "w") as f:
        json.dump(val_index, f, indent=2)

    with open(out_dir / "test_index.json", "w") as f:
        json.dump(test_index, f, indent=2)

    with open(out_dir / "demo_records.json", "w") as f:
        json.dump(demo_records, f, indent=2)

    with open(out_dir / "split_demo_records.json", "w") as f:
        json.dump(split_demo_records, f, indent=2)

    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print()
    print("Done.")
    print(f"Output dir: {out_dir}")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
