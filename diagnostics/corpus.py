"""Corpus access: the aligned prediction tables and their schema.

Every diagnostic starts here. `load` returns one dataset's table with one row
per positive test edge, `novel` masks the never-seen-before node pairs, and
`solved` turns a model column into a boolean solved set. Any column stem with
per-seed rank columns works as a model name, including ablated variants such
as "TPNet_nowalk".
"""
import os
import pandas as pd

CORPUS = os.path.join(os.path.dirname(__file__), "..", "corpus")
DATASETS = ["wikipedia", "reddit", "mooc", "uci", "uslegis", "canparl", "untrade", "enron"]
MODELS = ["TPNet", "DyGFormer", "TGN", "GraphMixer"]
SEEDS = [0, 1, 2]
K = 10

# the 18 interpretable edge descriptors (Table I of the paper). `recurs_later`
# is future-derived and only ever used to label edges, never as a feature;
# `pair_count`, `pair` recency (`time_since_last`) and `is_repeated` are
# constant on novel edges.
DESCRIPTORS = [
    "adamic_adar", "bipartite_cf_score", "both_new", "burst_density",
    "common_neighbors", "dst_new", "dst_popularity", "dst_recency",
    "is_repeated", "nbr_recency", "pair_count", "recurs_later", "res_alloc",
    "src_new", "src_popularity", "src_recency", "struct_proxy",
    "time_since_last",
]


def load(ds):
    """The aligned prediction table of one dataset (rows in chronological test order)."""
    return pd.read_csv(os.path.join(CORPUS, f"{ds}.csv.gz"))


def novel(tab):
    """Boolean mask of novel edges: node pairs never observed in training or validation."""
    return (tab["is_repeated"] == 0).to_numpy()


def solved(tab, model, seed, k=K):
    """Boolean solved set: rank of the true destination is at most k (hit@k)."""
    return (tab[f"{model}_s{seed}"] <= k).to_numpy()
