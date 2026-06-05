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
from phase_belief.models.phase_belief_gru import PhaseBeliefGRU


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-index",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_4files/train_index.json",
    )
    parser.add_argument(
        "--val-index",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/data_indices/libero_goal_4files/val_index.json",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="/root/autodl-tmp/phase_belief_libero/checkpoints/lowdim_phase_gru",
    )

    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    parser.add_argument("--input-dim", type=int, default=15)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--state-dim", type=int, default=8)
    parser.add_argument("--pred-len", type=int, default=4)
    parser.add_argument("--num-phases", type=int, default=6)
    parser.add_argument("--hidden-dim", type=int, default=128)

    parser.add_argument("--lambda-state", type=float, default=1.0)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)
    parser.add_argument("--lambda-balance", type=float, default=0.01)

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


def phase_stats(phase_probs):
    with torch.no_grad():
        mean_p = phase_probs.mean(dim=(0, 1))
        entropy = -(mean_p * torch.log(mean_p + 1e-8)).sum()
        max_frac = mean_p.max()
    return mean_p, entropy, max_frac


def run_one_epoch(model, loader, optimizer, device, args, train=True):
    if train:
        model.train()
    else:
        model.eval()

    total = {
        "loss": 0.0,
        "action": 0.0,
        "state": 0.0,
        "smooth": 0.0,
        "balance": 0.0,
        "n": 0,
    }

    phase_sum = None
    phase_count = 0

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

            loss = (
                action_loss
                + args.lambda_state * state_loss
                + args.lambda_smooth * smooth_loss
                + args.lambda_balance * balance_loss
            )

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

        bs = x.shape[0]
        total["loss"] += float(loss.item()) * bs
        total["action"] += float(action_loss.item()) * bs
        total["state"] += float(state_loss.item()) * bs
        total["smooth"] += float(smooth_loss.item()) * bs
        total["balance"] += float(balance_loss.item()) * bs
        total["n"] += bs

        with torch.no_grad():
            p = out["phase_probs"]
            p_sum = p.sum(dim=(0, 1)).detach().cpu()
            if phase_sum is None:
                phase_sum = p_sum
            else:
                phase_sum += p_sum
            phase_count += p.shape[0] * p.shape[1]

    metrics = {k: total[k] / max(total["n"], 1) for k in ["loss", "action", "state", "smooth", "balance"]}

    mean_p = phase_sum / max(phase_count, 1)
    entropy = -(mean_p * torch.log(mean_p + 1e-8)).sum().item()
    max_frac = mean_p.max().item()

    metrics["phase_entropy"] = entropy
    metrics["phase_max_frac"] = max_frac
    metrics["phase_mean"] = [round(float(v), 4) for v in mean_p.tolist()]

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

    model = PhaseBeliefGRU(
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

    config_path = save_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)

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
            f"phase max {val_metrics['phase_max_frac']:.3f} "
            f"phase mean {val_metrics['phase_mean']}"
        )

        last_ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "args": vars(args),
            "val_metrics": val_metrics,
        }

        torch.save(last_ckpt, save_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(last_ckpt, save_dir / "best.pt")

    train_set.close()
    val_set.close()

    print("done")
    print("best val loss:", best_val)
    print("save dir:", save_dir)


if __name__ == "__main__":
    main()
