#!/usr/bin/env python3
"""Fig. 4 of the paper: within-model ablations, read from the corpus.

(A) walk projections: TPNet vs TPNet_nowalk, novel edges, per configuration
(B) co-occurrence: DyGFormer vs DyGFormer_nocooc, novel and repeated
(C) time encoders: TPNet vs TPNet_lineartime, GraphMixer vs
    GraphMixer_learnedtime, novel edges
(D) convergence: change in Jaccard overlap between the ablated model's
    solved set and each competitor's
Also the descriptor-null check: descriptors do not separate lost from
retained edges consistently.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from diagnostics import load, novel, solved, jaccard, MODELS, SEEDS

def hits(tab, col, mask):
    return [solved(tab, col, s)[mask].mean() for s in SEEDS]

def fmt(v):
    return f"{np.mean(v):.3f}+-{np.std(v):.3f}"

print("(A) walk projections, novel edges")
for ds, full, abl in [("enron", "TPNet", "TPNet_nowalk"),
                      ("canparl", "TPNet", "TPNet_nowalk"),
                      ("canparl", "TPNet_recency", "TPNet_nowalk_recency")]:
    tab = load(ds); nov = novel(tab)
    print(f"  {ds:8s} {full:22s} {fmt(hits(tab, full, nov))}  -{'walk':6s} {fmt(hits(tab, abl, nov))}")

print("(B) co-occurrence")
for ds in ["uci", "canparl"]:
    tab = load(ds); nov = novel(tab)
    for stratum, mask in [("novel", nov), ("repeated", ~nov)]:
        print(f"  {ds:8s} {stratum:9s} DyGFormer {fmt(hits(tab, 'DyGFormer', mask))}  "
              f"-cooc {fmt(hits(tab, 'DyGFormer_nocooc', mask))}")

print("(C) time encoders, novel edges")
for ds in ["uci", "canparl"]:
    tab = load(ds); nov = novel(tab)
    print(f"  {ds:8s} TPNet      {fmt(hits(tab, 'TPNet', nov))}  linear    {fmt(hits(tab, 'TPNet_lineartime', nov))}")
    print(f"  {ds:8s} GraphMixer {fmt(hits(tab, 'GraphMixer', nov))}  learnable {fmt(hits(tab, 'GraphMixer_learnedtime', nov))}")

print("(D) convergence: delta Jaccard(ablated, competitor) - Jaccard(full, competitor), novel")
for ds, full, abl in [("enron", "TPNet", "TPNet_nowalk"), ("canparl", "TPNet", "TPNet_nowalk"),
                      ("uci", "DyGFormer", "DyGFormer_nocooc"), ("canparl", "DyGFormer", "DyGFormer_nocooc")]:
    tab = load(ds); nov = novel(tab)
    base = full.split("_")[0]
    out = []
    for comp in [m for m in MODELS if m != base]:
        deltas = []
        for s in SEEDS:
            Sc = solved(tab, comp, s)[nov]
            Sf = solved(tab, full, s)[nov]
            Sa = solved(tab, abl, s)[nov]
            deltas.append(jaccard(Sa, Sc) - jaccard(Sf, Sc))
        out.append(f"{comp} {np.mean(deltas):+.3f}")
    print(f"  {ds:8s} {abl:22s} " + "  ".join(out))

print("descriptor-null check: |mean(descriptor | lost) - mean(descriptor | retained)|,")
print("standardised; max over descriptors, per ablation (small + sign-flipping = no edge class)")
DESC = ["common_neighbors", "adamic_adar", "res_alloc", "bipartite_cf_score",
        "src_popularity", "dst_popularity", "src_recency", "dst_recency",
        "nbr_recency", "burst_density", "struct_proxy"]
for ds, full, abl in [("enron", "TPNet", "TPNet_nowalk"), ("canparl", "TPNet", "TPNet_nowalk"),
                      ("uci", "DyGFormer", "DyGFormer_nocooc"), ("canparl", "DyGFormer", "DyGFormer_nocooc")]:
    tab = load(ds); nov = novel(tab)
    zs = []
    for d in DESC:
        x = tab[d].to_numpy()[nov].astype(float)
        sd = x.std() + 1e-12
        vals = []
        for s in SEEDS:
            Sf = solved(tab, full, s)[nov]
            Sa = solved(tab, abl, s)[nov]
            lost, ret = Sf & ~Sa, Sf & Sa
            if lost.sum() and ret.sum():
                vals.append((x[lost].mean() - x[ret].mean()) / sd)
        zs.append((d, np.mean(vals)))
    top = max(zs, key=lambda t: abs(t[1]))
    print(f"  {ds:8s} {abl:22s} max |z| = {abs(top[1]):.2f} ({top[0]})")
