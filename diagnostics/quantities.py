"""Core quantities over solved sets (Sec. III-C of the paper).

All functions operate on boolean solved sets as produced by `corpus.solved`,
either stacked into a hit matrix H (one row per model, one column per edge) or
as a {(model, seed): mask} dictionary. They are deliberately small: a
diagnostic is a composition of these set operations, not a framework feature.
"""
import numpy as np

from .corpus import MODELS, SEEDS, K, novel, solved


def hit_matrix(tab, seed, models=MODELS, mask=None, k=K):
    """Stack solved sets into one row per model; optionally restrict to a mask."""
    H = np.vstack([solved(tab, m, seed, k) for m in models])
    return H[:, mask] if mask is not None else H


def solved_sets(tab, models=MODELS, seeds=SEEDS, mask=None, k=K):
    """{(model, seed): solved mask}, optionally restricted to a mask."""
    return {(m, s): (solved(tab, m, s, k)[mask] if mask is not None else solved(tab, m, s, k))
            for m in models for s in seeds}


def composition(H):
    """(all, contested, none) fractions of edges by how many models solve them."""
    nc = H.sum(0)
    n = H.shape[1]
    return ((nc == H.shape[0]).sum() / n,
            ((nc > 0) & (nc < H.shape[0])).sum() / n,
            (nc == 0).sum() / n)


def best_single(H):
    """Accuracy of the strongest single row."""
    return H.mean(1).max()


def oracle(H):
    """Accuracy of a per-edge oracle that picks any solving row."""
    return H.any(0).mean()


def drop_weakest_gain(H):
    """Oracle gain over best single after dropping the weakest row (control)."""
    singles = H.mean(1)
    keep = np.argsort(singles)[1:]
    return H[keep].any(0).mean() - singles[keep].max()


def pairwise_coverage(tab, models=MODELS, seeds=SEEDS, k=K):
    """Matrix M[i, j]: % of row i's correctly solved novel edges that column j misses."""
    nov = novel(tab)
    M = np.zeros((len(models), len(models)))
    for s in seeds:
        H = {m: solved(tab, m, s, k) & nov for m in models}
        for i, a in enumerate(models):
            for j, b in enumerate(models):
                if i != j:
                    M[i, j] += 100 * (H[a] & ~H[b]).sum() / max(H[a].sum(), 1)
    return M / len(seeds)


def jaccard(a, b):
    """Overlap of two solved sets."""
    return (a & b).sum() / max((a | b).sum(), 1)
