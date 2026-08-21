#!/usr/bin/env python3
"""Table VII of the paper: parameter-free score ensembles vs the learned gate.

Needs the per-candidate score files (see ../scores/README.md): for each
dataset, model and seed, an npz with the positive score and the 99 negative
scores of every test edge. Because the candidate slates are identical across
models, score-level fusion is exact.

  mean  mean of the models' sigmoid scores
  max   max of the sigmoid scores
  rank  mean within-slate rank (Borda)
  best  strongest single architecture (from the corpus, full novel set)
  gate  the learned fusion (from the corpus)

Usage: python3 ensembles.py [scores_dir]   (default: ../scores)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import (load, novel, load_scores, sigmoid, hit10, slate_ranks,
                         DATASETS, SEEDS, K)

SCORES_DIR = sys.argv[1] if len(sys.argv) > 1 else None
EXPERTS = ["TPNet", "DyGFormer", "TGN"]

print(f"{'dataset':10s} {'best':>13} {'mean':>13} {'max':>13} {'rank':>13} {'gate':>13}")
for ds in DATASETS:
    tab = load(ds)
    nov_by_ei = dict(zip(tab["edge_index"], tab["is_repeated"] == 0))
    res = {k: [] for k in ["best", "mean", "max", "rank", "gate"]}
    for s in SEEDS:
        data = {m: load_scores(ds, m, s, SCORES_DIR) for m in EXPERTS}
        e0 = data[EXPERTS[0]]["edge_index"]
        for m in EXPERTS[1:]:
            assert np.array_equal(e0, data[m]["edge_index"]), "edge order mismatch"
        nov = np.array([nov_by_ei[e] for e in e0])
        pos = np.stack([sigmoid(data[m]["pos"]) for m in EXPERTS])
        neg = np.stack([sigmoid(data[m]["neg"]) for m in EXPERTS])
        res["mean"].append(hit10(pos.mean(0), neg.mean(0))[nov].mean())
        res["max"].append(hit10(pos.max(0), neg.max(0))[nov].mean())
        rk = np.stack([slate_ranks(data[m]["pos"], data[m]["neg"]) for m in EXPERTS]).mean(0)
        res["rank"].append(hit10(-rk[:, 0], -rk[:, 1:])[nov].mean())
        t = tab.set_index("edge_index").loc[e0]
        novt = (t["is_repeated"] == 0).to_numpy()
        res["gate"].append((t[f"Fusion_s{s}"] <= K)[novt].mean())
        res["best"].append(max((t[f"{m}_s{s}"] <= K)[novt].mean()
                               for m in ["TPNet", "DyGFormer", "TGN", "GraphMixer"]))
    # 'best' in the paper is the strongest architecture by 3-seed mean
    tabm = load(ds); novm = novel(tabm)
    best = max(np.mean([np.asarray(tabm[f"{m}_s{s}"] <= K)[novm].mean() for s in SEEDS])
               for m in ["TPNet", "DyGFormer", "TGN", "GraphMixer"])
    cells = [f"{best:.3f}"] + [f"{np.mean(res[k]):.3f}+-{np.std(res[k]):.3f}"
                               for k in ["mean", "max", "rank", "gate"]]
    print(f"{ds:10s} " + " ".join(f"{c:>13}" for c in cells))
