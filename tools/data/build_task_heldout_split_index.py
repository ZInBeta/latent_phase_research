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
        "--out-root",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_task_heldout",
    )
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--expected-num-files", type=int, default=10)

    parser.add_argument(
        "--heldout-files",
        type=str,
        nargs="*",
        default=None,
        help="Held-out hdf5 file names or stems. If omitted, build leave-one-task-out splits for all hdf5 files.",
    )
    parser.add_argument("--overwrite", action="store_true")

    return parser.parse_args()


def demo_sort_key(name):
    try:
        return int(name.split("_")[-1])
    except Exception:
        return name


def normalize_file_key(name):
    name = Path(name).name
    if name.endswith(".hdf5"):
        return name
    return name + ".hdf5"


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
            "length": T,
            "start": start,
            "seq_len": seq_len,
            "pred_len": pred_len,
            "current_t": start + seq_len - 1,
            "future_start": start + seq_len,
            "future_end": start + seq_len + pred_len,
        })

    return windows


def split_train_val_per_file(records, val_ratio, seed):
    rng = random.Random(seed)

    by_file = {}
    for r in records:
        by_file.setdefault(r["file_name"], []).append(r)

    split_records = {
        "train": [],
        "val": [],
    }

    for file_name, file_records in sorted(by_file.items()):
        file_records = sorted(file_records, key=lambda x: demo_sort_key(x["demo_name"]))
        rng.shuffle(file_records)

        n = len(file_records)
        n_val = int(n * val_ratio)

        val_records = file_records[:n_val]
        train_records = file_records[n_val:]

        split_records["train"].extend(train_records)
        split_records["val"].extend(val_records)

        print(
            f"[Train/Val] {file_name}: "
            f"total={n}, train={len(train_records)}, val={len(val_records)}"
        )

    return split_records


def assert_no_demo_overlap(split_records):
    sets = {}

    for split, records in split_records.items():
        sets[split] = set((r["file_name"], r["demo_name"]) for r in records)

    split_names = list(sets.keys())
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            a = split_names[i]
            b = split_names[j]
            assert sets[a].isdisjoint(sets[b]), f"{a}/{b} demo overlap"


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def check_output_dir(out_dir, overwrite):
    existing_files = [
        "train_index.json",
        "val_index.json",
        "test_index.json",
        "config.json",
    ]

    if out_dir.exists() and not overwrite:
        if any((out_dir / name).exists() for name in existing_files):
            raise FileExistsError(
                f"Output dir already has split files: {out_dir}. "
                f"Use --overwrite if you want to regenerate it."
            )

    out_dir.mkdir(parents=True, exist_ok=True)


def build_one_split(args, heldout_file_name, demo_records, hdf5_files, out_dir):
    print()
    print("=" * 80)
    print(f"[Heldout] {heldout_file_name}")
    print(f"[Output]  {out_dir}")

    check_output_dir(out_dir, args.overwrite)

    usable_records = [r for r in demo_records if r["usable"]]

    test_records = [r for r in usable_records if r["file_name"] == heldout_file_name]
    train_val_records = [r for r in usable_records if r["file_name"] != heldout_file_name]

    if len(test_records) == 0:
        raise ValueError(f"No usable demos found for heldout file: {heldout_file_name}")

    train_val_split = split_train_val_per_file(
        train_val_records,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    split_records = {
        "train": train_val_split["train"],
        "val": train_val_split["val"],
        "test": test_records,
    }

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
        "data_dir": str(args.data_dir),
        "hdf5_files": [str(p) for p in hdf5_files],
        "num_hdf5_files": len(hdf5_files),
        "heldout_file": heldout_file_name,
        "seq_len": args.seq_len,
        "pred_len": args.pred_len,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "num_demos_total": len(demo_records),
        "num_demos_usable": len(usable_records),
        "num_demos_train": len(split_records["train"]),
        "num_demos_val": len(split_records["val"]),
        "num_demos_test": len(split_records["test"]),
        "num_windows_train": len(train_index),
        "num_windows_val": len(val_index),
        "num_windows_test": len(test_index),
        "split_rule": "task-held-out split: all demos from heldout hdf5 task file are test; remaining tasks are split into train/val at demo level.",
        "note": "states is not used for training input. This file only stores indices.",
    }

    write_json(out_dir / "train_index.json", train_index)
    write_json(out_dir / "val_index.json", val_index)
    write_json(out_dir / "test_index.json", test_index)
    write_json(out_dir / "demo_records.json", demo_records)
    write_json(out_dir / "split_demo_records.json", split_demo_records)
    write_json(out_dir / "config.json", config)

    print("[Done split]")
    print(json.dumps(config, indent=2))

    return config


def main():
    args = parse_args()

    if args.val_ratio <= 0.0 or args.val_ratio >= 1.0:
        raise ValueError("--val-ratio must be in (0, 1)")

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    args.data_dir = data_dir

    hdf5_files = sorted(data_dir.glob("*.hdf5"))
    if len(hdf5_files) != args.expected_num_files:
        raise ValueError(
            f"Expected {args.expected_num_files} hdf5 files, "
            f"but found {len(hdf5_files)} in {data_dir}"
        )

    hdf5_file_names = [p.name for p in hdf5_files]

    if args.heldout_files is None or len(args.heldout_files) == 0:
        heldout_files = hdf5_file_names
    else:
        heldout_files = [normalize_file_key(x) for x in args.heldout_files]

    missing = sorted(set(heldout_files) - set(hdf5_file_names))
    if missing:
        raise ValueError(f"Heldout files not found in data dir: {missing}")

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

    all_configs = []

    for heldout_file_name in heldout_files:
        out_dir = out_root / f"heldout_{Path(heldout_file_name).stem}"
        config = build_one_split(
            args=args,
            heldout_file_name=heldout_file_name,
            demo_records=demo_records,
            hdf5_files=hdf5_files,
            out_dir=out_dir,
        )
        all_configs.append(config)

    write_json(out_root / "all_heldout_splits_summary.json", all_configs)

    print()
    print("=" * 80)
    print("Done all task-heldout splits.")
    print(f"Output root: {out_root}")
    print(f"Num splits: {len(all_configs)}")


if __name__ == "__main__":
    main()