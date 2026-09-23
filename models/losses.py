# -*- coding: utf-8 -*-
"""
Model v0 objectives.

The diagnosis (PROBLEM_DIAGNOSIS.md v1.0, E6) says the failing quantity is NOT
the mean normal response -- `mean_defect` is uncorrelated with performance while
the upper tail of the normal response is strongly (negatively) correlated with
it. So the objective targets the tail, and a separate term stops the trivial
solution of collapsing every patch to one point (which would zero the tail and
the defect evidence alike).
"""
import torch
import torch.nn.functional as F


def tail_loss(dists, alpha=0.01):
    """CVaR-style upper-tail penalty: mean of the top-alpha of normal distances.

    `dists` are 1-NN distances of NORMAL query patches to the normal bank.
    alpha=0.01 mirrors the detector's own statistic (image score = mean of the
    top 1% of patch distances); larger alpha is a smoother, more stable target.
    """
    n = dists.numel()
    k = max(1, int(round(n * alpha)))
    return torch.topk(dists, k, largest=True).values.mean()


def mean_dist_loss(dists):
    """The obvious alternative objective -- average normal distance.

    This is the ablation's control (config B): it asks whether *adapting* helps,
    versus specifically targeting the tail.
    """
    return dists.mean()


def separation_loss(d_normal, d_defect, margin=0.05):
    """Push every DEFECT patch above the normal tail by `margin`.

    This is the objective the diagnosis *wants* to be zero: it is exactly
    `frac_defect_below_p99` in differentiable form. It requires defect labels,
    which the few-shot setting does not have -- hence its use only as an oracle
    probe, to separate "wrong target" from "missing information".

    The threshold is detached so the defect term chases a fixed target instead
    of the two terms jointly drifting downhill.
    """
    thr = torch.quantile(d_normal.detach(), 0.99)
    return torch.relu(thr + margin - d_defect).mean()


def preserve_loss(z_new, z_old):
    """Keep the pretrained geometry patch-by-patch.

    Without this the adapter can trivially zero the tail by mapping every patch
    to a single direction; the cost is that defect evidence vanishes too. This
    term penalises exactly that collapse.
    """
    return (1.0 - F.cosine_similarity(z_new, z_old, dim=-1)).mean()


def relation_loss(z_new, z_old, max_pairs=2048, gen=None):
    """Preserve pairwise similarity structure among a sample of patches.

    Optional and off by default: preserve_loss already blocks the degenerate
    collapse, and this is the term to reach for only if the adapter turns out to
    blur local structure while keeping per-patch cosine high.
    """
    n = z_new.shape[0]
    if n > max_pairs:
        idx = torch.randperm(n, device=z_new.device,
                             generator=gen)[:max_pairs]
        z_new, z_old = z_new[idx], z_old[idx]
    a = F.normalize(z_new.float(), dim=-1)
    b = F.normalize(z_old.float(), dim=-1)
    return ((a @ a.T) - (b @ b.T)).pow(2).mean()
