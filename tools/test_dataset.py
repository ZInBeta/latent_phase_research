from torch.utils.data import DataLoader

from phase_belief.data.dataset import PhaseBeliefDataset


def main():
    index_path = "/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_4files/train_index.json"

    dataset = PhaseBeliefDataset(index_path)

    print("num samples:", len(dataset))

    sample = dataset[0]

    print()
    print("single sample:")
    print("x:", sample["x"].shape)
    print("future_actions:", sample["future_actions"].shape)
    print("future_state_delta:", sample["future_state_delta"].shape)
    print("file_name:", sample["file_name"])
    print("demo_name:", sample["demo_name"])
    print("start:", sample["start"])
    print("current_t:", sample["current_t"])

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    batch = next(iter(loader))

    print()
    print("batch:")
    print("x:", batch["x"].shape)
    print("future_actions:", batch["future_actions"].shape)
    print("future_state_delta:", batch["future_state_delta"].shape)

    dataset.close()


if __name__ == "__main__":
    main()
