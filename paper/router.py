#!/usr/bin/env python3
"""Table VI of the paper plus the pairwise ownership probe.

Router: one random forest per architecture (200 trees, depth 6, min leaf 20)
predicts from the descriptors whether that model solves a novel edge; each edge
is routed to the architecture with the highest predicted success probability.
Trained on the chronological first half of the novel edges, evaluated on the
held-out second half.

Ownership probe: for each pair of architectures, restrict to edges exactly one
of the two solves (difficulty controlled by construction) and measure the AUC
of predicting which one from the descriptors. 0.5 = no ownership information.
"""
import itertools
import os
import sys

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import load, novel, solved_sets, DATASETS, MODELS, SEEDS, DESCRIPTORS

# usable router features: all descriptors except the label (`recurs_later`) and
# the ones constant on novel edges (`is_repeated`, `pair_count`, `time_since_last`).
# Kept as an explicit list because the forest is sensitive to column order and
# the paper numbers were produced with this one.
FEATS = ["src_popularity", "dst_popularity", "src_recency", "dst_recency",
         "common_neighbors", "bipartite_cf_score", "burst_density", "src_new",
         "dst_new", "both_new", "struct_proxy", "nbr_recency", "adamic_adar",
         "res_alloc"]
assert set(FEATS) == set(DESCRIPTORS) - {"recurs_later", "is_repeated",
                                         "pair_count", "time_since_last"}

def forest(seed):
    return RandomForestClassifier(n_estimators=200, max_depth=6,
                                  min_samples_leaf=20, random_state=seed, n_jobs=4)

print(f"{'dataset':10s} {'best':>6} {'router':>7} {'oracle':>7} {'recov':>7} {'ownAUC':>7}")
aucs_all = []
for ds in DATASETS:
    tab = load(ds); nov = novel(tab)
    X = np.column_stack([np.nan_to_num(tab[k].to_numpy()[nov].astype(float), nan=0.0)
                         for k in FEATS])
    n = X.shape[0]; cut = n // 2
    rows, aucs = [], []
    S = solved_sets(tab, mask=nov)
    for s in SEEDS:
        H = {m: S[m, s] for m in MODELS}
        # router on the held-out second half
        proba = np.zeros((n - cut, len(MODELS)))
        for j, m in enumerate(MODELS):
            y = H[m][:cut].astype(int)
            if y.sum() in (0, len(y)):
                proba[:, j] = y.mean()
            else:
                proba[:, j] = forest(s).fit(X[:cut], y).predict_proba(X[cut:])[:, 1]
        route = proba.argmax(1)
        Ht = np.column_stack([H[m][cut:] for m in MODELS])
        rows.append((Ht.mean(0).max(),
                     Ht[np.arange(len(route)), route].mean(),
                     Ht.max(1).mean()))
        # pairwise ownership AUC
        for a, b in itertools.combinations(MODELS, 2):
            xor = H[a] ^ H[b]; y = H[a].astype(int)
            tr_i = xor.copy(); tr_i[cut:] = False
            te_i = xor.copy(); te_i[:cut] = False
            if tr_i.sum() < 80 or te_i.sum() < 80: continue
            if len(set(y[tr_i])) < 2 or len(set(y[te_i])) < 2: continue
            clf = forest(s).fit(X[tr_i], y[tr_i])
            aucs.append(roc_auc_score(y[te_i], clf.predict_proba(X[te_i])[:, 1]))
    best, router, oracle = np.mean(rows, axis=0)
    gap = oracle - best
    rec = 100 * (router - best) / gap if gap > 1e-9 else float("nan")
    auc = np.mean(aucs) if aucs else float("nan")
    aucs_all.append(auc)
    print(f"{ds:10s} {best:6.3f} {router:7.3f} {oracle:7.3f} {rec:+6.0f}% {auc:7.3f}")
print(f"\nmean ownership AUC: {np.nanmean(aucs_all):.3f}")
