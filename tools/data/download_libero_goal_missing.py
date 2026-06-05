import os
import time
from pathlib import Path

import h5py
from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "yifengzhu-hf/LIBERO-datasets"
REPO_TYPE = "dataset"

LOCAL_ROOT = Path("/root/autodl-tmp/LIBERO/datasets")
LOCAL_GOAL_DIR = LOCAL_ROOT / "libero_goal"

MAX_RETRIES = 10
SLEEP_SECONDS = 20


def is_valid_hdf5(path: Path) -> bool:
    try:
        with h5py.File(path, "r") as f:
            return "data" in f and len(f["data"].keys()) > 0
    except Exception:
        return False


def main():
    os.environ["HF_HUB_DISABLE_XET"] = "1"
    os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "120"

    LOCAL_GOAL_DIR.mkdir(parents=True, exist_ok=True)

    print("Listing remote files...")
    all_files = list_repo_files(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
    )

    goal_files = sorted(
        f for f in all_files
        if f.startswith("libero_goal/") and f.endswith(".hdf5")
    )

    print(f"Remote libero_goal hdf5 files: {len(goal_files)}")
    for f in goal_files:
        print("  ", f)

    print()
    print("Checking local files...")
    missing_or_bad = []

    for remote_file in goal_files:
        local_path = LOCAL_ROOT / remote_file

        if local_path.exists() and is_valid_hdf5(local_path):
            print("OK   ", local_path.name)
        else:
            if local_path.exists():
                print("BAD  ", local_path.name)
            else:
                print("MISS ", local_path.name)
            missing_or_bad.append(remote_file)

    if not missing_or_bad:
        print()
        print("All libero_goal files are present and valid.")
        return

    print()
    print(f"Need to download: {len(missing_or_bad)} files")

    for remote_file in missing_or_bad:
        print()
        print("=" * 80)
        print("Downloading:", remote_file)

        last_err = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    filename=remote_file,
                    local_dir=str(LOCAL_ROOT),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

                local_path = LOCAL_ROOT / remote_file

                if is_valid_hdf5(local_path):
                    print("DONE ", local_path)
                    break

                raise RuntimeError(f"Downloaded file is not a valid hdf5: {local_path}")

            except Exception as e:
                last_err = e
                print(f"Attempt {attempt}/{MAX_RETRIES} failed:", repr(e))

                if attempt < MAX_RETRIES:
                    print(f"Sleep {SLEEP_SECONDS}s then retry...")
                    time.sleep(SLEEP_SECONDS)

        else:
            raise RuntimeError(f"Failed to download {remote_file}") from last_err

    print()
    print("Final validation:")
    ok_count = 0

    for remote_file in goal_files:
        local_path = LOCAL_ROOT / remote_file
        if is_valid_hdf5(local_path):
            with h5py.File(local_path, "r") as f:
                n = len(f["data"].keys())
            print("OK", local_path.name, "demos:", n)
            ok_count += 1
        else:
            print("BAD", local_path)

    print()
    print(f"Valid files: {ok_count}/{len(goal_files)}")
    print("Local dir:", LOCAL_GOAL_DIR)


if __name__ == "__main__":
    main()
