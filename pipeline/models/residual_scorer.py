import torch
import torch.nn as nn


class ResidualScorer(nn.Module):
    """
    Lightweight hard-edge residual scorer with optional low-history gate.

    Computes alpha (N,) and hard_logit (N,) from log1p-scaled history features.

    Final logit combination (in the training/eval loop):
        final_logit = tpnet_logit + alpha * hard_logit   [default]
        final_logit = hard_logit                          [--hard_only ablation]

    Gate types (--hard_gate_type):
      'learnable' : alpha = sigmoid(a - b * log1p(pair_count))   [default, 2 params]
      'exp'       : alpha = exp(-pair_count / tau)                [no params]
      'mlp'       : alpha = sigmoid(MLP_gate(full feature vec))
      'none'      : alpha = 1.0  (unconditional residual, ablation)

    Feature index 0 is expected to be log1p(pair_count) as produced by HistoryState.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 64,
                 gate_type: str = 'learnable', tau: float = 5.0):
        super().__init__()
        self.gate_type = gate_type
        self.tau = tau

        self.mlp_hard = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        if gate_type == 'learnable':
            # init: alpha≈0.88 for pair_count=0, ≈0.50 for pair_count≈1.7
            # raw_gate_b passes through softplus so b = softplus(raw_b) >= 0 always
            self.gate_a = nn.Parameter(torch.tensor(2.0))
            self.raw_gate_b = nn.Parameter(torch.tensor(0.5413))  # softplus(0.5413)≈1.0
        elif gate_type == 'mlp':
            self.mlp_gate = nn.Sequential(
                nn.Linear(feat_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

    def forward(self, features: torch.Tensor):
        """
        Args:
            features: (N, feat_dim) float tensor of log1p-scaled history features.
                      features[:, 0] must be log1p(pair_count).
        Returns:
            alpha: (N,)  gate in [0, 1]
            hard_logit: (N,)
        """
        hard_logit = self.mlp_hard(features).squeeze(-1)
        log_pc = features[:, 0]

        if self.gate_type == 'exp':
            pair_count = torch.expm1(log_pc).clamp(min=0.0)
            alpha = torch.exp(-pair_count / self.tau)
        elif self.gate_type == 'learnable':
            b = torch.nn.functional.softplus(self.raw_gate_b)
            alpha = torch.sigmoid(self.gate_a - b * log_pc)
        elif self.gate_type == 'mlp':
            alpha = torch.sigmoid(self.mlp_gate(features).squeeze(-1))
        else:  # 'none'
            alpha = torch.ones(features.shape[0], device=features.device,
                               dtype=features.dtype)
        return alpha, hard_logit
