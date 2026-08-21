#!/usr/bin/env python3
"""Fig. 3 of the paper: novel-edge hit@10 per architecture and dataset
(3-seed mean +/- std), with the EdgeBank memorisation baseline."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import load, novel, solved, DATASETS, MODELS, SEEDS, K

print(f"{'dataset':10s} " + " ".join(f"{m:>16}" for m in MODELS) + f" {'EdgeBank':>9}")
for ds in DATASETS:
    tab = load(ds); nov = novel(tab)
    cells = []
    for m in MODELS:
        h = [solved(tab, m, s)[nov].mean() for s in SEEDS]
        cells.append(f"{np.mean(h):.3f}+-{np.std(h):.3f}")
    eb = (tab["EdgeBank"][nov] <= K).mean()
    print(f"{ds:10s} " + " ".join(f"{c:>16}" for c in cells) + f" {eb:9.3f}")
