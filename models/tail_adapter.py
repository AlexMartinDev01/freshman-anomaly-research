# -*- coding: utf-8 -*-
"""
Model v0 backbone-side module: a residual adapter on frozen DINOv2 patch tokens.

    z' = z + fc2(relu(fc1(z)))          z in R^384

The output layer is zero-initialised, so training starts *exactly* at the
frozen-feature baseline -- every change the adapter makes is a deliberate
departure from DINOv2, which keeps the A/B/C/D ablation interpretable (a random
initialisation would already perturb the tail before any loss is applied).

The backbone is never trained. Only this module is.
"""
import torch
import torch.nn as nn


class ResidualAdapter(nn.Module):
    def __init__(self, dim=384, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        return z + self.fc2(torch.relu(self.fc1(z)))

    def extra_repr(self):
        return f"dim={self.fc1.in_features}, hidden={self.fc1.out_features}"


def l2norm(x, eps=1e-8):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def nn_distances(query, bank, chunk=512):
    """Per-query 1-NN cosine distance to `bank` (both L2-normalised here).

    Returns 1 - max cosine similarity, matching AnomalyDINO's scoring exactly.
    Chunked over the query so the (chunk x bank) matrix stays small -- and so
    that autograd does not hold one giant (query x bank) matrix alive.

    Differentiable w.r.t. both arguments; wrap eval calls in torch.no_grad().
    """
    q, b = l2norm(query.float()), l2norm(bank.float())
    out = []
    for i in range(0, q.shape[0], chunk):
        sim = q[i:i + chunk] @ b.T              # (chunk, bank)
        out.append(1.0 - sim.max(dim=1).values)
    return torch.cat(out)
