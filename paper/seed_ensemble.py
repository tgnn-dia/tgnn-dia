#!/usr/bin/env python3
"""Seed-ensemble control (Sec. VI-B): is a combination gain variance reduction
or architectural complementarity?

Fuses the three seeds of ONE architecture (no architectural diversity) and
compares against the cross-architecture fusion of ensembles.py. If the
seed ensemble reproduces the gain, variance reduction suffices; if not, the
gain needs genuinely different architectures.

Usage: python3 seed_ensemble.py [scores_dir]   (default: ../scores)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import load, load_scores, sigmoid, hit10, slate_ranks, DATASETS, SEEDS

SCORES_DIR = sys.argv[1] if len(sys.argv) > 1 else None

print(f"{'dataset':10s} {'model':10s} {'singles':>15} {'seed-mean':>9} {'seed-rank':>9}")
for ds in DATASETS:
    tab = load(ds)
    nov_by_ei = dict(zip(tab["edge_index"], tab["is_repeated"] == 0))
    for m in ["TPNet", "DyGFormer", "TGN"]:
        data = {s: load_scores(ds, m, s, SCORES_DIR) for s in SEEDS}
        e0 = data[0]["edge_index"]; nov = np.array([nov_by_ei[e] for e in e0])
        singles = [hit10(data[s]["pos"], data[s]["neg"])[nov].mean() for s in SEEDS]
        pos = np.stack([sigmoid(data[s]["pos"]) for s in SEEDS])
        neg = np.stack([sigmoid(data[s]["neg"]) for s in SEEDS])
        mean_h = hit10(pos.mean(0), neg.mean(0))[nov].mean()
        rk = np.stack([slate_ranks(data[s]["pos"], data[s]["neg"]) for s in SEEDS]).mean(0)
        rank_h = hit10(-rk[:, 0], -rk[:, 1:])[nov].mean()
        print(f"{ds:10s} {m:10s} {np.mean(singles):7.3f}+-{np.std(singles):.3f} "
              f"{mean_h:9.3f} {rank_h:9.3f}")
