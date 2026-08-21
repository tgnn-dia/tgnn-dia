"""Per-candidate score access and score-level fusion primitives.

The corpus stores one rank per model and edge. For score-level combination the
release additionally provides, per dataset, model and seed, the raw score of
the positive and of each of the 99 negatives of every test edge (see
scores/README.md). Because the candidate slates are identical across models
and seeds, fusing scores and re-ranking is exact, no re-evaluation needed.
"""
import os
import numpy as np

from .corpus import K

SCORES = os.path.join(os.path.dirname(__file__), "..", "scores")


def load_scores(ds, model, seed, scores_dir=None):
    """npz with `edge_index`, `pos` (n,), `neg` (n, 99) raw scores of one run."""
    return np.load(os.path.join(scores_dir or SCORES, f"{ds}_{model}_call{seed}.npz"))


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def hit10(pos, neg, k=K):
    """Per-edge hit@k of positive scores against their negative slates."""
    return (1 + (neg > pos[:, None]).sum(1)) <= k


def slate_ranks(pos, neg):
    """Within-slate rank of every candidate (0 = highest score), per edge."""
    all_s = np.concatenate([pos[:, None], neg], axis=1)
    order = np.argsort(-all_s, axis=1, kind="stable")
    rk = np.empty_like(order)
    rk[np.arange(len(all_s))[:, None], order] = np.arange(all_s.shape[1])[None, :]
    return rk
