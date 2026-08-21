#!/usr/bin/env python3
"""Tables II-IV of the paper: separating architecture from initialisation.

Table II  disagreement |S_a xor S_b|/|A| between architectures (fixed seed)
          vs between seeds (fixed architecture), novel edges
Table III per-edge oracle over the best three different architectures
          (same seed) vs over three seeds of the best single architecture
Table IV  largest set one architecture solves at all three seeds while
          another misses at all three
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import load, novel, solved_sets, DATASETS, MODELS, SEEDS

print(f"{'dataset':10s} {'between-arch%':>14} {'between-seed%':>14} "
      f"{'3seed-oracle':>13} {'3arch-oracle':>13} {'improv':>7} {'stable%':>8}")
for ds in DATASETS:
    tab = load(ds); nov = novel(tab)
    S = solved_sets(tab, mask=nov)
    # Table II
    arch = [np.mean(S[a, s] ^ S[b, s]) for s in SEEDS
            for a, b in itertools.combinations(MODELS, 2)]
    seed = [np.mean(S[m, s1] ^ S[m, s2]) for m in MODELS
            for s1, s2 in itertools.combinations(SEEDS, 2)]
    # Table III: both columns are the best over their choices of a per-edge
    # oracle across three runs (three seeds of one model vs three architectures
    # at one seed, averaged over seeds)
    seed_oracle = max(np.logical_or.reduce([S[m, s] for s in SEEDS]).mean()
                      for m in MODELS)
    # the trio is chosen by coverage, not by individual accuracy: on some
    # datasets it keeps an individually weaker model for its complementary edges
    trios = {trio: np.mean([np.logical_or.reduce([S[m, s] for m in trio]).mean()
                            for s in SEEDS])
             for trio in itertools.combinations(MODELS, 3)}
    best_trio = max(trios, key=trios.get)
    arch_oracle = trios[best_trio]
    dropped = (set(MODELS) - set(best_trio)).pop()
    # Table IV
    stable = 0
    for a, b in itertools.permutations(MODELS, 2):
        owned = np.logical_and.reduce([S[a, s] & ~S[b, s] for s in SEEDS])
        stable = max(stable, owned.mean())
    print(f"{ds:10s} {100*np.mean(arch):14.1f} {100*np.mean(seed):14.1f} "
          f"{seed_oracle:13.3f} {arch_oracle:13.3f} {100*(arch_oracle-seed_oracle):+7.1f} "
          f"{100*stable:8.1f}   trio drops {dropped}")
