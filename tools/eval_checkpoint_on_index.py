import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from phase_belief.data.dataset import PhaseBeliefDataset
from phase_belief.models.phase_belief_bottleneck_gru import PhaseBeliefBottleneckGRU


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--index", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    return parser.parse_args()


def smooth_loss_fn(phase_probs):
    diff = phase_probs[:, 1:] - phase_probs[:, :-1]
    return (diff ** 2).mean()


def balance_loss_fn(phase_probs, eps=1e-8):
    num_phases = phase_probs.shape[-1]
    mean_p = phase_probs.mean(dim=(0, 1))
    uniform = torch.full_like(mean_p, 1.0 / num_phases)
    return (mean_p * (torch.log(mean_p + eps) - torch.log(uniform + eps))).sum()


def confidence_loss_fn(phase_probs, eps=1e-8):
    entropy = -(phase_probs * torch.log(phase_probs + eps)).sum(dim=-1)
    return entropy.mean()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt["args"]

    model = PhaseBeliefBottleneckGRU(
        input_dim=cfg["input_dim"],
        action_dim=cfg["action_dim"],
        state_dim=cfg["state_dim"],
        pred_len=cfg["pred_len"],
        num_phases=cfg["num_phases"],
        hidden_dim=cfg["hidden_dim"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    dataset = PhaseBeliefDataset(args.index)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    total = {
        "loss": 0.0,
        "action": 0.0,
        "state": 0.0,
        "smooth": 0.0,
        "balance": 0.0,
        "conf": 0.0,
        "n": 0,
    }

    phase_sum = None
    phase_count = 0
    inst_entropy_sum = 0.0
    inst_max_sum = 0.0
    num_batches = 0

    lambda_state = cfg.get("lambda_state", 1.0)
    lambda_smooth = cfg.get("lambda_smooth", 0.05)
    lambda_balance = cfg.get("lambda_balance", 0.02)
    lambda_conf = cfg.get("lambda_conf", 0.01)

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target_actions = batch["future_actions"].to(device, non_blocking=True)
        target_state_delta = batch["future_state_delta"].to(device, non_blocking=True)

        out = model(x)

        action_loss = F.mse_loss(out["pred_actions"], target_actions)
        state_loss = F.mse_loss(out["pred_state_delta"], target_state_delta)
        smooth_loss = smooth_loss_fn(out["phase_probs"])
        balance_loss = balance_loss_fn(out["phase_probs"])
        conf_loss = confidence_loss_fn(out["phase_probs"])

        loss = (
            action_loss
            + lambda_state * state_loss
            + lambda_smooth * smooth_loss
            + lambda_balance * balance_loss
            + lambda_conf * conf_loss
        )

        bs = x.shape[0]

        total["loss"] += float(loss.item()) * bs
        total["action"] += float(action_loss.item()) * bs
        total["state"] += float(state_loss.item()) * bs
        total["smooth"] += float(smooth_loss.item()) * bs
        total["balance"] += float(balance_loss.item()) * bs
        total["conf"] += float(conf_loss.item()) * bs
        total["n"] += bs

        p = out["phase_probs"]
        p_sum = p.sum(dim=(0, 1)).detach().cpu()

        if phase_sum is None:
            phase_sum = p_sum
        else:
            phase_sum += p_sum

        phase_count += p.shape[0] * p.shape[1]

        inst_entropy = -(p * torch.log(p + 1e-8)).sum(dim=-1).mean().item()
        inst_max = p.max(dim=-1).values.mean().item()

        inst_entropy_sum += inst_entropy
        inst_max_sum += inst_max
        num_batches += 1

    metrics = {
        k: total[k] / max(total["n"], 1)
        for k in ["loss", "action", "state", "smooth", "balance", "conf"]
    }

    mean_usage = phase_sum / max(phase_count, 1)
    usage_entropy = -(mean_usage * torch.log(mean_usage + 1e-8)).sum().item()

    metrics["num_samples"] = total["n"]
    metrics["phase_usage_entropy"] = usage_entropy
    metrics["phase_inst_entropy"] = inst_entropy_sum / max(num_batches, 1)
    metrics["phase_inst_max_prob"] = inst_max_sum / max(num_batches, 1)
    metrics["phase_mean"] = [round(float(v), 6) for v in mean_usage.tolist()]
    metrics["ckpt"] = args.ckpt
    metrics["index"] = args.index

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))

    dataset.close()


if __name__ == "__main__":
    main()
