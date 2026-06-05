import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve()
for parent in PROJECT_ROOT.parents:
    if (parent / "phase_belief").is_dir():
        sys.path.insert(0, str(parent))
        break

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from phase_belief.data.dataset import PhaseBeliefDataset
from phase_belief.models.phase_belief_bottleneck_gru import PhaseBeliefBottleneckGRU


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train-index", type=str, default="/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_4files/train_index.json")
    parser.add_argument("--val-index", type=str, default="/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_4files/val_index.json")
    parser.add_argument("--save-dir", type=str, default="/root/autodl-tmp/phase_belief_libero/checkpoints/lowdim_phase_bottleneck_k4")

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--input-dim", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--num-phases", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-balance", type=float, default=0.02)
    parser.add_argument("--lambda-conf", type=float, default=0.01)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def collect_phase_metrics(phase_probs):
    with torch.no_grad():
        mean_usage = phase_probs.mean(dim=(0, 1))
        usage_entropy = -(mean_usage * torch.log(mean_usage + 1e-8)).sum().item()
        inst_entropy = -(phase_probs * torch.log(phase_probs + 1e-8)).sum(dim=-1).mean().item()
        inst_max_prob = phase_probs.max(dim=-1).values.mean().item()

    return mean_usage.detach().cpu(), usage_entropy, inst_entropy, inst_max_prob


def run_one_epoch(model, loader, optimizer, device, args, train=True):
    model.train() if train else model.eval()

    totals = {
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
    batch_count = 0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        target_actions = batch["future_actions"].to(device, non_blocking=True)
        target_state_delta = batch["future_state_delta"].to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            out = model(x)

            action_loss = F.mse_loss(out["pred_actions"], target_actions)
            state_loss = F.mse_loss(out["pred_state_delta"], target_state_delta)
            smooth_loss = smooth_loss_fn(out["phase_probs"])
            balance_loss = balance_loss_fn(out["phase_probs"])
            conf_loss = confidence_loss_fn(out["phase_probs"])

            loss = (
                action_loss
                + args.lambda_state * state_loss
                + args.lambda_smooth * smooth_loss
                + args.lambda_balance * balance_loss
                + args.lambda_conf * conf_loss
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = x.shape[0]
        totals["loss"] += float(loss.item()) * bs
        totals["action"] += float(action_loss.item()) * bs
        totals["state"] += float(state_loss.item()) * bs
        totals["smooth"] += float(smooth_loss.item()) * bs
        totals["balance"] += float(balance_loss.item()) * bs
        totals["conf"] += float(conf_loss.item()) * bs
        totals["n"] += bs

        mean_usage, _, inst_entropy, inst_max_prob = collect_phase_metrics(out["phase_probs"])

        if phase_sum is None:
            phase_sum = mean_usage * bs
        else:
            phase_sum += mean_usage * bs

        phase_count += bs
        inst_entropy_sum += inst_entropy
        inst_max_sum += inst_max_prob
        batch_count += 1

    metrics = {k: totals[k] / max(totals["n"], 1) for k in ["loss", "action", "state", "smooth", "balance", "conf"]}

    mean_usage = phase_sum / max(phase_count, 1)
    usage_entropy = -(mean_usage * torch.log(mean_usage + 1e-8)).sum().item()

    metrics["phase_usage_entropy"] = usage_entropy
    metrics["phase_inst_entropy"] = inst_entropy_sum / max(batch_count, 1)
    metrics["phase_inst_max_prob"] = inst_max_sum / max(batch_count, 1)
    metrics["phase_mean"] = [round(float(v), 4) for v in mean_usage.tolist()]

    return metrics


def main():
    args = parse_args()
    set_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_set = PhaseBeliefDataset(args.train_index)
    val_set = PhaseBeliefDataset(args.val_index)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = PhaseBeliefBottleneckGRU(
        input_dim=args.input_dim,
        action_dim=args.action_dim,
        state_dim=args.state_dim,
        pred_len=args.pred_len,
        num_phases=args.num_phases,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    save_args = vars(args).copy()
    save_args["model_type"] = "phase_bottleneck_gru"

    with open(save_dir / "config.json", "w") as f:
        json.dump(save_args, f, indent=2)

    log_path = save_dir / "train_log.jsonl"
    best_val = math.inf

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, args, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer, device, args, train=False)

        row = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }

        with open(log_path, "a") as f:
            f.write(json.dumps(row) + "\n")

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.6f} "
            f"act {train_metrics['action']:.6f} "
            f"state {train_metrics['state']:.6f} | "
            f"val loss {val_metrics['loss']:.6f} "
            f"act {val_metrics['action']:.6f} "
            f"state {val_metrics['state']:.6f} | "
            f"inst_max {val_metrics['phase_inst_max_prob']:.3f} "
            f"inst_ent {val_metrics['phase_inst_entropy']:.3f} "
            f"usage {val_metrics['phase_mean']}"
        )

        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": save_args,
            "val_metrics": val_metrics,
        }

        torch.save(ckpt, save_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, save_dir / "best.pt")

    train_set.close()
    val_set.close()

    print("done")
    print("best val loss:", best_val)
    print("save dir:", save_dir)


if __name__ == "__main__":
    main()
