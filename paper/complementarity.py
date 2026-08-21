#!/usr/bin/env python3
"""Fig. 2 of the paper: outcome composition, oracle payoff, pairwise coverage.

Per dataset, on novel edges (3-seed mean):
  all4 / contested / none  fractions, best single, best-of-4 oracle, oracle
  gain, and the drop-weakest control. Then the pairwise coverage matrix
  (share of row's solved edges the column misses) for one dataset.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import (load, novel, hit_matrix, composition, best_single,
                         oracle, drop_weakest_gain, pairwise_coverage,
                         DATASETS, MODELS, SEEDS)

print(f"{'dataset':10s} {'all4%':>6} {'cont%':>6} {'none%':>6} {'best':>6} "
      f"{'oracle':>7} {'gain':>6} {'drop-weakest gain':>18}")
for ds in DATASETS:
    tab = load(ds); nov = novel(tab)
    a, c, z, b, o, dwg = [], [], [], [], [], []
    for s in SEEDS:
        H = hit_matrix(tab, s, mask=nov)
        al, co, no = composition(H)
        a.append(al); c.append(co); z.append(no)
        b.append(best_single(H)); o.append(oracle(H))
        dwg.append(drop_weakest_gain(H))
    print(f"{ds:10s} {100*np.mean(a):6.1f} {100*np.mean(c):6.1f} {100*np.mean(z):6.1f} "
          f"{np.mean(b):6.3f} {np.mean(o):7.3f} {100*(np.mean(o)-np.mean(b)):+6.1f} "
          f"{100*np.mean(dwg):+18.1f}")

ds = sys.argv[1] if len(sys.argv) > 1 else "canparl"
M = pairwise_coverage(load(ds))
print(f"\npairwise coverage on {ds} (% of row's correct novel edges the column misses):")
print("            " + " ".join(f"{m[:9]:>10}" for m in MODELS))
for i, m in enumerate(MODELS):
    print(f"{m:11s} " + " ".join("         -" if i == j else f"{M[i,j]:10.0f}" for j in range(4)))
