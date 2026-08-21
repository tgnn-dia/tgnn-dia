#!/usr/bin/env python3
"""Evaluate EdgeBank (non-learnable memorization baseline) under the SAME
1-vs-99 deterministic-historical-negative protocol as the trained models, and
save per-edge ranks to saved_results/v2_link_{ds}_EdgeBank_seed0_test_predictions.csv
so it slots into the per-edge analysis. CPU-only; does not touch the GPU runs.

Tie convention: rank = 1 + #{neg_score >= pos_score} (pessimistic, matches the MRR
ClassicEvaluator). For EdgeBank's binary scores this is essential — strict '>'
would credit 0-vs-0 ties on novel edges as rank-1 'solves', which is wrong."""
import sys, csv, logging, numpy as np
sys.path.insert(0, ".")
logger = logging.getLogger("x"); logging.basicConfig(level=logging.ERROR)
from utils.DataLoader import get_link_prediction_data_classic, Data
from models.EdgeBank import edge_bank_link_prediction

BATCH = 200
TEST_RATIO = 0.15
# EdgeBank per-dataset config for HISTORICAL negatives (from the original TPNet/DyGLib repo)
CFG = {  # ds: (memory_mode, time_window_mode)
    "uci": ("time_window_memory","fixed_proportion"),
    "canparl": ("time_window_memory","fixed_proportion"),
    "uslegis": ("time_window_memory","fixed_proportion"),
    "mooc": ("time_window_memory","repeat_interval"),
    "lastfm": ("time_window_memory","repeat_interval"),
    "enron": ("time_window_memory","repeat_interval"),
    "untrade": ("time_window_memory","repeat_interval"),
    "unvote": ("time_window_memory","repeat_interval"),
    "contacts": ("time_window_memory","repeat_interval"),
    "wikipedia": ("repeat_threshold_memory","fixed_proportion"),
    "reddit": ("repeat_threshold_memory","fixed_proportion"),
    "socialevo": ("repeat_threshold_memory","fixed_proportion"),
    "flights": ("repeat_threshold_memory","fixed_proportion"),
}

def run(ds):
    mem_mode, win_mode = CFG[ds]
    r = get_link_prediction_data_classic(ds, "historical", logger)
    train, val, test, sampler = r[3], r[4], r[5], r[6]
    tv_src = np.concatenate([train.src_node_ids, val.src_node_ids])
    tv_dst = np.concatenate([train.dst_node_ids, val.dst_node_ids])
    tv_t   = np.concatenate([train.node_interact_times, val.node_interact_times])
    tv_e   = np.concatenate([train.edge_ids, val.edge_ids])
    tv_l   = np.concatenate([train.labels, val.labels])
    n = len(test.src_node_ids); records = []
    for b0 in range(0, n, BATCH):
        b1 = min(b0 + BATCH, n)
        bs, bd = test.src_node_ids[b0:b1], test.dst_node_ids[b0:b1]
        bt = test.node_interact_times[b0:b1]
        # growing history = train+val + test seen before this batch (matches original EdgeBank eval)
        hist = Data(src_node_ids=np.concatenate([tv_src, test.src_node_ids[:b0]]),
                    dst_node_ids=np.concatenate([tv_dst, test.dst_node_ids[:b0]]),
                    node_interact_times=np.concatenate([tv_t, test.node_interact_times[:b0]]),
                    edge_ids=np.concatenate([tv_e, test.edge_ids[:b0]]),
                    labels=np.concatenate([tv_l, test.labels[:b0]]))
        # SAME deterministic 99 negatives as the trained models
        neg = sampler.query_batch(pos_src=bs - 1, pos_dst=bd - 1, pos_timestamp=bt, split_mode="test")
        neg = (np.array(neg, dtype=bs.dtype) + 1)            # (B, 99)
        nneg = neg.shape[1]
        neg_src = np.repeat(bs, nneg); neg_dst = neg.reshape(-1)
        pos_p, neg_p = edge_bank_link_prediction(history_data=hist, positive_edges=(bs, bd),
                                                 negative_edges=(neg_src, neg_dst),
                                                 edge_bank_memory_mode=mem_mode, time_window_mode=win_mode,
                                                 time_window_proportion=TEST_RATIO)
        neg_p = neg_p.reshape(len(bs), nneg)
        # midpoint (expected) tie-break: coincides with strict ranking for the
        # continuous-score models (no ties); fair for EdgeBank's binary scores.
        gt = (neg_p > pos_p[:, None]).sum(axis=1)
        eq = (neg_p == pos_p[:, None]).sum(axis=1)
        ranks = 1.0 + gt + 0.5 * eq
        for j in range(len(bs)):
            records.append((b0 + j, float(ranks[j])))
    out = f"./saved_results/v2_link_{ds}_EdgeBank_seed0_test_predictions.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["edge_index", "rank"]); w.writerows(records)
    # quick repeated/novel sanity: mean rank
    rk = np.array([x[1] for x in records])
    print(f"{ds:10s} mem={mem_mode[:6]}/{win_mode[:5]}  n={n}  hit@10={np.mean(rk<=10):.3f}  "
          f"mrr={np.mean(1/rk):.3f}  -> {out.split('/')[-1]}", flush=True)

if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["uslegis","uci","canparl","untrade","wikipedia"]
    for ds in datasets:
        run(ds)
