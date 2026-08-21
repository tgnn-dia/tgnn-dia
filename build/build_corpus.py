#!/usr/bin/env python3
"""Assemble the aligned prediction corpus from raw per-run prediction files.

This script documents provenance. It uses the dataset loaders and descriptor
code from ../pipeline and needs the raw per-run prediction dumps (RAW_RESULTS);
the released corpus/*.csv.gz files are its output, and every script
in ../paper reads only those files.

One table per dataset. One row per positive test edge. Columns:
  edge_index, src, dst, time      identifiers (chronological test order)
  <18 descriptor columns>         computed from training+validation history
                                  only, except `recurs_later` (future-derived,
                                  used to label edges, never as a descriptor)
  <Model>_s<seed>                 rank of the true destination among 1+99
                                  deterministic, identical candidates (int)
  EdgeBank                        deterministic baseline, fractional ranks
                                  from expected mid-rank tie-breaking
Ablation columns exist only for the datasets they were run on.
"""
import os, sys, gzip, logging
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "pipeline"))
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("build")
from analyze_edge_predictions import build_history, compute_features
from utils.DataLoader import get_link_prediction_data_classic

# raw per-run prediction csv dumps written by the training pipeline
# (one file per dataset x model x seed; not part of this repository)
RES = os.environ.get("RAW_RESULTS", os.path.join(REPO, "saved_results"))
OUT = os.path.join(REPO, "corpus")
DS = ["wikipedia", "reddit", "mooc", "uci", "uslegis", "canparl", "untrade", "enron"]
SEEDS = [0, 1, 2]
CORE = {"TPNet": "v2", "DyGFormer": "v2", "TGN": "v2", "GraphMixer": "v2", "MyModel": "v2"}
COLNAME = {"MyModel": "Fusion"}  # historical internal name of the fusion model in raw run files
# ablated variants, per dataset: column stem -> (prefix, model)
ABL = {
    "enron":   {"TPNet_nowalk": ("v2norp", "TPNet")},
    "canparl": {"TPNet_nowalk": ("v2norp", "TPNet"),
                "TPNet_recency": ("recfull", "TPNet"),
                "TPNet_nowalk_recency": ("ablnorp", "TPNet"),
                "DyGFormer_nocooc": ("ablcooc", "DyGFormer"),
                "GraphMixer_learnedtime": ("abllearn", "GraphMixer"),
                "TPNet_lineartime": ("abllin", "TPNet")},
    "uci":     {"DyGFormer_nocooc": ("ablcooc", "DyGFormer"),
                "GraphMixer_learnedtime": ("abllearn", "GraphMixer"),
                "TPNet_lineartime": ("abllin", "TPNet")},
}

def ranks(prefix, ds, model, seed, ei):
    f = f"{RES}/{prefix}_link_{ds}_{model}_seed{seed}_test_predictions.csv"
    return pd.read_csv(f).set_index("edge_index").reindex(ei)["rank"].to_numpy()

for ds in DS:
    _, _, _, tr, va, te, _, _ = get_link_prediction_data_classic(ds, "historical", logger)
    sh, dh, th = build_history(tr, va)
    ref = pd.read_csv(f"{RES}/v2_link_{ds}_TPNet_seed0_test_predictions.csv").sort_values("edge_index")
    ei = ref["edge_index"].to_numpy()
    feats = compute_features(te.src_node_ids[ei], te.dst_node_ids[ei],
                             te.node_interact_times[ei], sh, dh, th,
                             update_with_test_history=False)
    tab = pd.DataFrame({"edge_index": ei,
                        "src": te.src_node_ids[ei],
                        "dst": te.dst_node_ids[ei],
                        "time": te.node_interact_times[ei]})
    for k in sorted(feats):
        tab[k] = feats[k]
    for model, prefix in CORE.items():
        col = COLNAME.get(model, model)
        for s in SEEDS:
            tab[f"{col}_s{s}"] = ranks(prefix, ds, model, s, ei).astype(int)
    tab["EdgeBank"] = ranks("v2", ds, "EdgeBank", 0, ei)
    for col, (prefix, model) in ABL.get(ds, {}).items():
        for s in SEEDS:
            tab[f"{col}_s{s}"] = ranks(prefix, ds, model, s, ei).astype(int)
    out = f"{OUT}/{ds}.csv.gz"
    tab.to_csv(out, index=False, compression="gzip", float_format="%.6g")
    print(f"{ds}: {len(tab)} rows, {len(tab.columns)} cols -> {out}")
print("done")
