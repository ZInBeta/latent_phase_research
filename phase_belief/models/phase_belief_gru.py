import torch
import torch.nn as nn
import torch.nn.functional as F


class PhaseBeliefGRU(nn.Module):
    def __init__(
        self,
        input_dim=15,
        action_dim=7,
        state_dim=8,
        pred_len=4,
        num_phases=6,
        hidden_dim=128,
        num_layers=1,
        dropout=0.1,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.pred_len = pred_len
        self.num_phases = num_phases
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.phase_head = nn.Linear(hidden_dim, num_phases)

        self.phase_embed = nn.Parameter(torch.randn(num_phases, hidden_dim) * 0.02)

        pred_in_dim = hidden_dim * 2 + num_phases

        self.pred_trunk = nn.Sequential(
            nn.Linear(pred_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.action_head = nn.Linear(hidden_dim, pred_len * action_dim)
        self.state_head = nn.Linear(hidden_dim, pred_len * state_dim)

    def forward(self, x):
        """
        x: [B, T, input_dim]
        """
        B = x.shape[0]

        h = self.input_proj(x)
        h_seq, _ = self.gru(h)

        phase_logits = self.phase_head(h_seq)
        phase_probs = F.softmax(phase_logits, dim=-1)

        h_last = h_seq[:, -1]
        p_last = phase_probs[:, -1]

        phase_context = p_last @ self.phase_embed

        pred_feat = torch.cat([h_last, phase_context, p_last], dim=-1)
        pred_feat = self.pred_trunk(pred_feat)

        pred_actions = self.action_head(pred_feat).view(B, self.pred_len, self.action_dim)
        pred_state_delta = self.state_head(pred_feat).view(B, self.pred_len, self.state_dim)

        return {
            "phase_logits": phase_logits,
            "phase_probs": phase_probs,
            "pred_actions": pred_actions,
            "pred_state_delta": pred_state_delta,
        }
