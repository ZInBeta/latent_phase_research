import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PhaseBeliefDataset(Dataset):
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)

        with open(self.index_path, "r") as f:
            self.index = json.load(f)

        self._h5_cache = {}

    def __len__(self):
        return len(self.index)

    def _get_h5(self, file_path: str):
        if file_path not in self._h5_cache:
            self._h5_cache[file_path] = h5py.File(file_path, "r")
        return self._h5_cache[file_path]

    def __getitem__(self, idx):
        item = self.index[idx]

        file_path = item["file_path"]
        demo_name = item["demo_name"]

        start = int(item["start"])
        seq_len = int(item["seq_len"])
        current_t = int(item["current_t"])
        future_start = int(item["future_start"])
        future_end = int(item["future_end"])

        f = self._get_h5(file_path)
        demo = f["data"][demo_name]
        obs = demo["obs"]

        ee_pos_seq = obs["ee_pos"][start:start + seq_len].astype(np.float32)
        ee_ori_seq = obs["ee_ori"][start:start + seq_len].astype(np.float32)
        gripper_seq = obs["gripper_states"][start:start + seq_len].astype(np.float32)

        action_dim = demo["actions"].shape[-1]
        prev_action_seq = np.zeros((seq_len, action_dim), dtype=np.float32)

        if seq_len > 1:
            prev_action_seq[1:] = demo["actions"][start:start + seq_len - 1].astype(np.float32)

        state_seq = np.concatenate(
            [ee_pos_seq, ee_ori_seq, gripper_seq],
            axis=-1,
        )

        x = np.concatenate(
            [state_seq, prev_action_seq],
            axis=-1,
        ).astype(np.float32)

        future_actions = demo["actions"][future_start:future_end].astype(np.float32)

        current_ee_pos = obs["ee_pos"][current_t].astype(np.float32)
        current_ee_ori = obs["ee_ori"][current_t].astype(np.float32)
        current_gripper = obs["gripper_states"][current_t].astype(np.float32)
        current_state = np.concatenate(
            [current_ee_pos, current_ee_ori, current_gripper],
            axis=-1,
        ).astype(np.float32)

        future_ee_pos = obs["ee_pos"][future_start:future_end].astype(np.float32)
        future_ee_ori = obs["ee_ori"][future_start:future_end].astype(np.float32)
        future_gripper = obs["gripper_states"][future_start:future_end].astype(np.float32)
        future_states = np.concatenate(
            [future_ee_pos, future_ee_ori, future_gripper],
            axis=-1,
        ).astype(np.float32)

        future_state_delta = future_states - current_state[None, :]

        return {
            "x": torch.from_numpy(x),
            "future_actions": torch.from_numpy(future_actions),
            "future_state_delta": torch.from_numpy(future_state_delta),
            "file_name": item["file_name"],
            "demo_name": demo_name,
            "start": start,
            "current_t": current_t,
        }

    def close(self):
        for h5_file in self._h5_cache.values():
            h5_file.close()
        self._h5_cache = {}
