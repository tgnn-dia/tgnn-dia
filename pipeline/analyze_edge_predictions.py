#!/usr/bin/env python3
"""
Edge-regime diagnosis for temporal link prediction models.

Core output: rank-bin characterisation of test edges — which edge regimes are
rank-1, which are near misses, and which form the long-tail failures.

Usage (single model):
    python analyze_edge_predictions.py \
        --predictions saved_results/std_link_tgbl-wiki_TPNet_seed0_test_predictions.csv \
        --dataset_name tgbl-wiki --rank_bins 1,10,100 \
        --out_dir analysis_results/std_tgbl-wiki

Two-model comparison:
    python analyze_edge_predictions.py \
        --predictions  saved_results/modelA_..._test_predictions.csv \
        --predictions2 saved_results/modelB_..._test_predictions.csv \
        --model_name TPNet --model2_name DyGFormer \
        --dataset_name tgbl-wiki --out_dir analysis_results/compare
"""

import argparse
import bisect
import json
import logging
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.tree import DecisionTreeClassifier, export_text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.DataLoader import (
    get_link_prediction_data, get_link_prediction_data_classic, CLASSIC_DATASETS,
)

logger = logging.getLogger(__name__)


# ── history building ──────────────────────────────────────────────────────────

def build_history(train_data, val_data):
    """Merge train+val edges, sort chronologically."""
    srcs  = np.concatenate([train_data.src_node_ids,        val_data.src_node_ids])
    dsts  = np.concatenate([train_data.dst_node_ids,        val_data.dst_node_ids])
    times = np.concatenate([train_data.node_interact_times, val_data.node_interact_times])
    order = np.argsort(times, kind='stable')
    return srcs[order], dsts[order], times[order]


# ── feature computation ───────────────────────────────────────────────────────

# Features derived from FUTURE information — valid for stratified analysis only,
# must never enter a surrogate / model-facing feature set (would be label leakage).
LEAKY_FEATURES = {'recurs_later'}


def compute_features(src_ids, dst_ids, test_times,
                     srcs_hist, dsts_hist, times_hist,
                     update_with_test_history: bool = False):
    """
    Compute per-edge diagnostic features aligned with the test edges.

    Features:
      is_repeated          1 if (src,dst) appeared in history
      pair_count           # prior (src,dst) occurrences
      time_since_last      seconds since last (src,dst) occurrence; NaN for unseen pairs
      src_popularity       # interactions by src (out-degree)
      dst_popularity       # interactions to dst (in-degree)
      src_recency          time since src was last active in any role
      dst_recency          time since dst was last active in any role
      common_neighbors     undirected common-neighbour count (0 for bipartite datasets)
      bipartite_cf_score   # co-visitors of dst who share ≥1 item with src (CF-style)
      burst_density        edge rate in adaptive time window around the test event
    """
    n = len(src_ids)

    # static history indices
    pair_count_map  = defaultdict(int)
    pair_last_map   = {}
    dst_count_map   = defaultdict(int)
    src_count_map   = defaultdict(int)
    node_last_map   = {}
    out_nbrs        = defaultdict(set)   # src → dsts
    in_nbrs         = defaultdict(set)   # dst → srcs

    for s, d, t in zip(srcs_hist, dsts_hist, times_hist):
        si, di, ti = int(s), int(d), float(t)
        pair_count_map[(si, di)] += 1
        pair_last_map[(si, di)]   = ti
        dst_count_map[di]        += 1
        src_count_map[si]        += 1
        node_last_map[si] = max(node_last_map.get(si, -1e18), ti)
        node_last_map[di] = max(node_last_map.get(di, -1e18), ti)
        out_nbrs[si].add(di)
        in_nbrs[di].add(si)

    hist_times_sorted = list(np.sort(times_hist))
    time_range  = float(hist_times_sorted[-1] - hist_times_sorted[0]) if len(hist_times_sorted) > 1 else 1.0
    burst_window = max(time_range * 0.001, 1.0)

    # running copies for streaming mode
    if update_with_test_history:
        pc_run  = dict(pair_count_map)
        pl_run  = dict(pair_last_map)
        dc_run  = dict(dst_count_map)
        sc_run  = dict(src_count_map)
        nl_run  = dict(node_last_map)
        on_run  = {k: set(v) for k, v in out_nbrs.items()}
        inn_run = {k: set(v) for k, v in in_nbrs.items()}
        t_run   = list(hist_times_sorted)
    else:
        pc_run  = pair_count_map
        pl_run  = pair_last_map
        dc_run  = dst_count_map
        sc_run  = src_count_map
        nl_run  = node_last_map
        on_run  = out_nbrs
        inn_run = in_nbrs
        t_run   = hist_times_sorted

    # output arrays
    a_is_rep   = np.zeros(n, np.float32)
    a_pcount   = np.zeros(n, np.float32)
    a_tsl      = np.full(n, np.nan, np.float64)   # time_since_last — NaN for unseen
    a_srcpop   = np.zeros(n, np.float32)
    a_dstpop   = np.zeros(n, np.float32)
    a_srcrec   = np.full(n, np.nan, np.float64)
    a_dstrec   = np.full(n, np.nan, np.float64)
    a_comnbr   = np.zeros(n, np.float32)
    a_cfsc     = np.zeros(n, np.float32)
    a_burst    = np.zeros(n, np.float32)
    a_src_new   = np.zeros(n, np.float32)   # src unseen in history (inductive/cold)
    a_dst_new   = np.zeros(n, np.float32)   # dst unseen in history (inductive/cold)
    a_both_new  = np.zeros(n, np.float32)   # both endpoints unseen
    a_struct_proxy = np.zeros(n, np.float32)  # 1=prior edge, 2=shared neighbour, 3=long-range
    # surrounding-structure / temporal-context features
    a_nbr_rec   = np.full(n, np.nan, np.float64)  # recency of freshest common neighbour (small=fresh structure)
    a_aa        = np.zeros(n, np.float32)   # Adamic-Adar proximity over common neighbours
    a_ra        = np.zeros(n, np.float32)   # resource-allocation proximity over common neighbours
    a_recurs    = np.zeros(n, np.float32)   # ANALYSIS-ONLY (uses future): pair recurs later in this stream

    MAX_CF = 100  # cap src item neighbourhood for CF to avoid O(n²) on popular nodes

    for i in range(n):
        src = int(src_ids[i])
        dst = int(dst_ids[i])
        t   = float(test_times[i])

        # pair recurrence
        cnt = pc_run.get((src, dst), 0)
        a_pcount[i]  = float(cnt)
        a_is_rep[i]  = 1.0 if cnt > 0 else 0.0
        if cnt > 0:
            a_tsl[i] = t - pl_run[(src, dst)]

        # popularity
        a_srcpop[i] = float(sc_run.get(src, 0))
        a_dstpop[i] = float(dc_run.get(dst, 0))

        # node recency
        if src in nl_run:
            a_srcrec[i] = t - nl_run[src]
        if dst in nl_run:
            a_dstrec[i] = t - nl_run[dst]

        # cold-start / inductive flags: node unseen in history up to this edge
        s_new = 0.0 if src in nl_run else 1.0
        d_new = 0.0 if dst in nl_run else 1.0
        a_src_new[i]  = s_new
        a_dst_new[i]  = d_new
        a_both_new[i] = 1.0 if (s_new > 0 and d_new > 0) else 0.0

        # common neighbours (undirected; always 0 for bipartite datasets)
        nbrs_s = on_run.get(src, set()) | inn_run.get(src, set())
        nbrs_d = on_run.get(dst, set()) | inn_run.get(dst, set())
        cmn    = nbrs_s & nbrs_d
        a_comnbr[i] = float(len(cmn))

        # surrounding structure: weighted proximity over common neighbours
        # (Adamic-Adar / resource-allocation down-weight high-degree hubs) and the
        # freshness of that structure (recency of the most recently active shared neighbour).
        if cmn:
            last_act = -1e18
            aa = 0.0
            ra = 0.0
            for w in cmn:
                dw = len(on_run.get(w, set()) | inn_run.get(w, set()))
                if dw > 1:
                    aa += 1.0 / np.log(dw)
                if dw > 0:
                    ra += 1.0 / dw
                lw = nl_run.get(w, -1e18)
                if lw > last_act:
                    last_act = lw
            a_aa[i] = aa
            a_ra[i] = ra
            a_nbr_rec[i] = t - last_act

        # bipartite CF: co-visitors of dst that share ≥1 item with src
        src_items = on_run.get(src, set())
        dst_visitors = inn_run.get(dst, set())
        if src_items and dst_visitors:
            capped_items = list(src_items)[:MAX_CF]
            covisitors = set()
            for item in capped_items:
                covisitors.update(inn_run.get(item, set()))
            covisitors.discard(src)
            a_cfsc[i] = float(len(dst_visitors & covisitors))

        # structural-distance proxy (long-range need): 1 = prior direct (src,dst) edge
        # (memorizable); 2 = shares a neighbour / co-visitor (2-hop shortcut); 3 = no
        # prior edge and no structural shortcut (long-range / hardest).
        if cnt > 0:
            a_struct_proxy[i] = 1.0
        elif a_comnbr[i] > 0 or a_cfsc[i] > 0:
            a_struct_proxy[i] = 2.0
        else:
            a_struct_proxy[i] = 3.0

        # burst density
        lo = bisect.bisect_left(t_run, t - burst_window)
        hi = bisect.bisect_right(t_run, t)
        a_burst[i] = float(hi - lo) / burst_window

        # streaming update (after reading, before next edge)
        if update_with_test_history:
            pc_run[(src, dst)] = cnt + 1
            pl_run[(src, dst)] = t
            dc_run[dst] = dc_run.get(dst, 0) + 1
            sc_run[src] = sc_run.get(src, 0) + 1
            nl_run[src] = max(nl_run.get(src, -1e18), t)
            nl_run[dst] = max(nl_run.get(dst, -1e18), t)
            on_run.setdefault(src, set()).add(dst)
            inn_run.setdefault(dst, set()).add(src)
            bisect.insort(t_run, t)

    # ANALYSIS-ONLY future-context: does this (src,dst) pair recur later in the
    # analysed stream? (an "emerging" relationship vs a one-off). Uses future
    # information by construction → never feed to a surrogate / model (see LEAKY_FEATURES).
    seen_after = set()
    for i in range(n - 1, -1, -1):
        key = (int(src_ids[i]), int(dst_ids[i]))
        if key in seen_after:
            a_recurs[i] = 1.0
        seen_after.add(key)

    # fill NaN recency (node never seen before) with 2× the max observed gap
    for arr in (a_srcrec, a_dstrec, a_nbr_rec):
        finite = np.isfinite(arr)
        fill = arr[finite].max() * 2.0 if finite.any() else 1.0
        arr[~finite] = fill

    return {
        'is_repeated':        a_is_rep,
        'pair_count':         a_pcount,
        'time_since_last':    a_tsl,        # NaN for unseen — kept raw
        'src_popularity':     a_srcpop,
        'dst_popularity':     a_dstpop,
        'src_recency':        a_srcrec.astype(np.float32),
        'dst_recency':        a_dstrec.astype(np.float32),
        'common_neighbors':   a_comnbr,
        'bipartite_cf_score': a_cfsc,
        'burst_density':      a_burst,
        'src_new':            a_src_new,
        'dst_new':            a_dst_new,
        'both_new':           a_both_new,
        'struct_proxy':       a_struct_proxy,
        'nbr_recency':        a_nbr_rec.astype(np.float32),
        'adamic_adar':        a_aa,
        'res_alloc':          a_ra,
        'recurs_later':       a_recurs,   # ANALYSIS-ONLY (future) — excluded from attribution
    }


# ── degenerate feature check ─────────────────────────────────────────────────

def check_degenerate(features):
    """Warn about and return the set of degenerate (constant / all-NaN) feature names."""
    bad = set()
    for name, vals in features.items():
        finite = vals[np.isfinite(vals)]
        if len(finite) == 0:
            logger.warning(f"Feature '{name}': all NaN — excluded from KS/tree")
            bad.add(name)
        elif np.std(finite) == 0:
            logger.warning(f"Feature '{name}': constant ({finite[0]:.4g}) — excluded from KS/tree")
            bad.add(name)
    return bad


# ── rank bins ────────────────────────────────────────────────────────────────

def parse_rank_bins(s: str):
    """
    '1,10,100' → [(1,1,'rank=1'), (2,10,'2≤rank≤10'), (11,100,'11≤rank≤100'), (101,None,'rank>100')]
    """
    thresholds = sorted(int(x) for x in s.split(','))
    bins, prev = [], 1
    for thr in thresholds:
        label = f'rank={prev}' if prev == thr else f'{prev}≤rank≤{thr}'
        bins.append((prev, thr, label))
        prev = thr + 1
    bins.append((prev, None, f'rank>{thresholds[-1]}'))
    return bins


def assign_rank_bin(ranks: np.ndarray, bins: list) -> np.ndarray:
    labels = np.empty(len(ranks), dtype=object)
    for lo, hi, label in bins:
        mask = (ranks >= lo) if hi is None else (ranks >= lo) & (ranks <= hi)
        labels[mask] = label
    return labels


# ── rank-bin summary table ────────────────────────────────────────────────────

def rank_bin_summary(ranks, bin_labels, features):
    rr = 1.0 / ranks.astype(float)
    rows = []
    for label in dict.fromkeys(bin_labels):   # preserves insertion order
        m = bin_labels == label
        n = m.sum()
        if n == 0:
            continue

        rep_m  = m & (features['is_repeated'] == 1)
        tsl    = features['time_since_last']
        tsl_ok = rep_m & np.isfinite(tsl)

        row = {
            'rank_bin':      label,
            'n':             int(n),
            '%all':          f"{100*n/len(ranks):.1f}",
            'MRR':           f"{rr[m].mean():.4f}",
            'rep_rate':      f"{features['is_repeated'][m].mean():.3f}",
            'pair_cnt_mean': f"{features['pair_count'][m].mean():.2f}",
            'pair_cnt_med':  f"{np.median(features['pair_count'][m]):.0f}",
            'tsl_mean(rep)': f"{tsl[tsl_ok].mean():.3g}" if tsl_ok.any() else 'n/a',
            'tsl_med(rep)':  f"{np.median(tsl[tsl_ok]):.3g}" if tsl_ok.any() else 'n/a',
            'src_pop_mean':  f"{features['src_popularity'][m].mean():.1f}",
            'dst_pop_mean':  f"{features['dst_popularity'][m].mean():.1f}",
            'src_rec_mean':  f"{features['src_recency'][m].mean():.3g}",
            'dst_rec_mean':  f"{features['dst_recency'][m].mean():.3g}",
            'burst_mean':    f"{features['burst_density'][m].mean():.3g}",
        }
        rows.append(row)
    return pd.DataFrame(rows)


# ── KS tests ─────────────────────────────────────────────────────────────────

def ks_tests(features, labels):
    """
    Two-sample KS test for each feature between labels==1 and labels==0.
    time_since_last is tested only on repeated pairs (where it is not NaN).
    """
    results = {}
    for name, vals in features.items():
        if name == 'time_since_last':
            rep = features.get('is_repeated')
            mask = (rep == 1) & np.isfinite(vals) if rep is not None else np.isfinite(vals)
        else:
            mask = np.isfinite(vals)
        v = vals[mask];  lb = labels[mask]
        succ = v[lb == 1];  fail = v[lb == 0]
        if len(succ) < 5 or len(fail) < 5:
            continue
        stat, pval = stats.ks_2samp(succ, fail)
        results[name] = dict(
            ks_stat=float(stat), p_value=float(pval),
            mean_succ=float(succ.mean()), mean_fail=float(fail.mean()),
            median_succ=float(np.median(succ)), median_fail=float(np.median(fail)),
            n_succ=int(len(succ)), n_fail=int(len(fail)),
        )
    return results


def fmt_ks(ks_res):
    lines = []
    for feat, r in sorted(ks_res.items(), key=lambda x: -x[1]['ks_stat']):
        sig = "***" if r['p_value'] < 0.001 else ("**" if r['p_value'] < 0.01 else
              ("*"  if r['p_value'] < 0.05  else "   "))
        lines.append(
            f"  {feat:<24} KS={r['ks_stat']:.3f}{sig} "
            f"| succ {r['mean_succ']:.3g} (med {r['median_succ']:.3g}) "
            f"| fail {r['mean_fail']:.3g} (med {r['median_fail']:.3g}) "
            f"| n={r['n_succ']}+{r['n_fail']}"
        )
    return "\n".join(lines) if lines else "  (no valid features)"


# ── bucketed MRR ─────────────────────────────────────────────────────────────

def bucketed_mrr(feature_vals, ranks, n_bins=5):
    """MRR within each quantile bin of a feature. Returns list of dicts or None."""
    rr = 1.0 / ranks.astype(float)
    finite = np.isfinite(feature_vals)
    if finite.sum() < n_bins * 2:
        return None
    v, r = feature_vals[finite], rr[finite]
    edges = np.percentile(v, np.linspace(0, 100, n_bins + 1))
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (v >= lo) & (v <= hi) if i == n_bins - 1 else (v >= lo) & (v < hi)
        if m.sum() == 0:
            continue
        rows.append({'bin': f'[{lo:.3g},{hi:.3g})', 'n': int(m.sum()), 'mrr': float(r[m].mean())})
    return rows


def fmt_bmrr(all_bmrr):
    lines = []
    for feat, rows in all_bmrr.items():
        if not rows:
            continue
        lines.append(f"  {feat}:")
        for r in rows:
            lines.append(f"    {r['bin']:>22s}  n={r['n']:>5d}  MRR={r['mrr']:.4f}")
    return "\n".join(lines)


# ── decision tree ─────────────────────────────────────────────────────────────

def tree_analysis(features, labels, max_depth=4):
    """
    Fit a DecisionTreeClassifier for binary regime labels.
    time_since_last NaN is filled with 2× max for tree input only.
    Returns (tree_text, importances_dict, metrics_dict).
    """
    tree_feats = {}
    for name, vals in features.items():
        if name == 'time_since_last':
            finite = np.isfinite(vals)
            fill = vals[finite].max() * 2.0 if finite.any() else 1.0
            tree_feats[name] = np.where(finite, vals, fill).astype(np.float32)
        else:
            tree_feats[name] = vals.copy()

    feat_names = [k for k, v in tree_feats.items() if np.std(v) > 0]
    if not feat_names:
        return "No non-degenerate features.", {}, {}

    X   = np.column_stack([tree_feats[k] for k in feat_names])
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=0)
    clf.fit(X, labels)
    pred       = clf.predict(X)
    pred_proba = clf.predict_proba(X)[:, 1]

    metrics = {
        'accuracy':          float((pred == labels).mean()),
        'balanced_accuracy': float(balanced_accuracy_score(labels, pred)),
        'precision_fail':    float(precision_score(labels, pred, pos_label=0, zero_division=0)),
        'recall_fail':       float(recall_score(labels, pred, pos_label=0, zero_division=0)),
    }
    try:
        metrics['auroc'] = float(roc_auc_score(labels, pred_proba))
    except ValueError:
        metrics['auroc'] = float('nan')

    importances = {n: float(v) for n, v in zip(feat_names, clf.feature_importances_)}
    return export_text(clf, feature_names=feat_names), importances, metrics


# ── N-model intersection analysis ────────────────────────────────────────────

def multi_model_comparison(model_dfs: dict, af: dict, edge_indices: np.ndarray,
                           K: int, hard_fail_rank: int) -> tuple[str, dict]:
    """
    Given N model DataFrames (name → DataFrame with edge_index+rank),
    aligned to the same edge_index set, produce:
      - universal success / universal failure groups
      - model-specific success patterns
      - all pairwise contingency tables
      - feature KS tests for discriminative edge regimes
    Returns (report_str, results_dict).
    """
    model_names = list(model_dfs.keys())
    M = len(model_names)

    # ── align all models to the reference edge set ────────────────────────────
    # edge_indices is the reference (from the first / primary model)
    model_ranks: dict[str, np.ndarray] = {}
    for name, df in model_dfs.items():
        ei = df['edge_index'].to_numpy()
        rk = df['rank'].to_numpy()
        # keep only edges present in reference
        ei_to_rk = {int(e): int(r) for e, r in zip(ei, rk)}
        aligned = np.array([ei_to_rk.get(int(e), 0) for e in edge_indices], dtype=np.int32)
        missing = (aligned == 0).sum()
        if missing > 0:
            logger.warning(f"  {name}: {missing} edges missing from reference set — assigned rank=0")
        model_ranks[name] = np.maximum(aligned, 1)  # rank≥1

    N = len(edge_indices)
    # success matrix: (N, M), True = rank ≤ K
    suc_mat = np.stack([(model_ranks[m] <= K) for m in model_names], axis=1)  # (N, M)
    hf_mat  = np.stack([(model_ranks[m] >  hard_fail_rank) for m in model_names], axis=1)

    all_suc  = suc_mat.all(axis=1)   # every model succeeds
    any_suc  = suc_mat.any(axis=1)   # at least one model succeeds
    all_fail = (~suc_mat).all(axis=1)

    feat_names = list(af.keys())
    X = np.column_stack([af[k] for k in feat_names])

    # ── helper: feature summary for a boolean mask ───────────────────────────
    def feat_summary(mask):
        if mask.sum() == 0:
            return {}
        return {k: {'mean': float(np.nanmean(X[mask, j])),
                    'median': float(np.nanmedian(X[mask, j]))}
                for j, k in enumerate(feat_names)}

    # ── helper: KS tests between two boolean masks ───────────────────────────
    def group_ks(mask_a, mask_b, min_n=10):
        if mask_a.sum() < min_n or mask_b.sum() < min_n:
            return {}
        res = {}
        for j, k in enumerate(feat_names):
            a = X[mask_a, j][np.isfinite(X[mask_a, j])]
            b = X[mask_b, j][np.isfinite(X[mask_b, j])]
            if len(a) < 5 or len(b) < 5:
                continue
            stat, p = stats.ks_2samp(a, b)
            res[k] = {'ks_stat': float(stat), 'p_value': float(p)}
        return res

    lines = [
        f"\n{'═'*60}",
        f"MULTI-MODEL COMPARISON  ({M} models | N={N} | K={K} | hard_fail>{hard_fail_rank})",
        f"{'─'*60}",
    ]

    # ── per-model headline ────────────────────────────────────────────────────
    lines.append("Per-model performance:")
    per_model_out = {}
    for name in model_names:
        r  = model_ranks[name]
        sr = float((r <= K).mean())
        r1 = float((r == 1).mean())
        hf = float((r > hard_fail_rank).mean())
        mrr = float(np.mean(1.0 / r))
        lines.append(f"  {name:<22}  MRR={mrr:.4f}  rank≤{K}={sr:.3f}  "
                     f"rank=1={r1:.3f}  hard_fail={hf:.3f}")
        per_model_out[name] = dict(mrr=mrr, success_rate=sr, rank1_rate=r1, hard_fail_rate=hf)

    # ── universal groups ──────────────────────────────────────────────────────
    lines += ["", "Universal groups:"]
    n_all_suc  = int(all_suc.sum())
    n_all_fail = int(all_fail.sum())
    n_any_suc  = int(any_suc.sum())
    lines.append(f"  all-success   (rank≤{K} for every model): n={n_all_suc}  "
                 f"({100*n_all_suc/N:.1f}%)")
    lines.append(f"  all-fail      (rank>{K} for every model): n={n_all_fail}  "
                 f"({100*n_all_fail/N:.1f}%)")
    lines.append(f"  any-success   (rank≤{K} for ≥1 model):    n={n_any_suc}  "
                 f"({100*n_any_suc/N:.1f}%)")

    group_features_out = {
        'all_success': feat_summary(all_suc),
        'all_fail':    feat_summary(all_fail),
    }
    ks_suc_vs_fail = group_ks(all_suc, all_fail)

    if ks_suc_vs_fail:
        lines += ["", "  KS: universal-success vs universal-fail features:"]
        for k, v in sorted(ks_suc_vs_fail.items(), key=lambda x: -x[1]['ks_stat']):
            sig = '***' if v['p_value'] < 0.001 else ('**' if v['p_value'] < 0.01 else
                  ('*' if v['p_value'] < 0.05 else ''))
            fs = af[k]
            ms = float(np.nanmean(fs[all_suc])) if all_suc.any() else float('nan')
            mf = float(np.nanmean(fs[all_fail])) if all_fail.any() else float('nan')
            lines.append(f"    {k:<24}  KS={v['ks_stat']:.3f}{sig}  "
                         f"succ_mean={ms:.2f}  fail_mean={mf:.2f}")

    # ── model-unique successes ────────────────────────────────────────────────
    lines += ["", "Model-unique successes (model_i rank≤K, all others rank>K):"]
    unique_out = {}
    for i, name in enumerate(model_names):
        other_idx = [j for j in range(M) if j != i]
        mask = suc_mat[:, i] & (~suc_mat[:, other_idx]).all(axis=1)
        n = int(mask.sum())
        lines.append(f"  {name:<22}  n={n}  ({100*n/N:.2f}%)")
        entry: dict = {'count': n, 'rate': float(n / N)}
        if n >= 10:
            entry['feature_means'] = feat_summary(mask)
            # DT: can we predict which edges this model uniquely solves?
            y_u = mask.astype(int)
            if len(np.unique(y_u)) > 1:
                X_tree = np.where(np.isfinite(X), X, 0.0)
                dt = DecisionTreeClassifier(max_depth=4, random_state=0)
                dt.fit(X_tree, y_u)
                entry['decision_tree'] = export_text(dt, feature_names=feat_names)
                entry['dt_accuracy'] = float(dt.score(X_tree, y_u))
        unique_out[name] = entry

    # ── model-unique failures ─────────────────────────────────────────────────
    lines += ["", "Model-unique failures (model_i rank>hard_fail, all others rank≤K):"]
    unique_fail_out = {}
    for i, name in enumerate(model_names):
        other_idx = [j for j in range(M) if j != i]
        mask = hf_mat[:, i] & suc_mat[:, other_idx].all(axis=1)
        n = int(mask.sum())
        lines.append(f"  {name:<22}  n={n}  ({100*n/N:.2f}%)")
        unique_fail_out[name] = {'count': n, 'rate': float(n / N)}

    # ── pairwise contingency table ────────────────────────────────────────────
    lines += ["", "Pairwise success/fail overlap (rank≤K):"]
    hdr = f"  {'':22}"
    for name in model_names:
        hdr += f"  {name[:10]:>10}"
    lines.append(hdr)

    pairwise_out = {}
    for i, ni in enumerate(model_names):
        row = f"  {ni:<22}"
        for j, nj in enumerate(model_names):
            if i == j:
                row += f"  {'—':>10}"
            else:
                both  = int((suc_mat[:, i] & suc_mat[:, j]).sum())
                row += f"  {both:>10}"
        lines.append(row)
    lines.append(f"  (cell = both models rank≤{K})")

    # detailed pairwise stats
    lines += ["", "Pairwise detailed:"]
    for i in range(M):
        for j in range(i + 1, M):
            ni, nj = model_names[i], model_names[j]
            both_s  = int((suc_mat[:, i] & suc_mat[:, j]).sum())
            i_only  = int((suc_mat[:, i] & ~suc_mat[:, j]).sum())
            j_only  = int((~suc_mat[:, i] & suc_mat[:, j]).sum())
            both_f  = int((~suc_mat[:, i] & ~suc_mat[:, j]).sum())
            ks = group_ks(suc_mat[:, i] & ~suc_mat[:, j],
                          ~suc_mat[:, i] & suc_mat[:, j])
            top_feat = max(ks.items(), key=lambda x: x[1]['ks_stat'])[0] if ks else 'n/a'
            top_ks   = ks[top_feat]['ks_stat'] if ks else 0.0
            lines.append(f"  {ni} vs {nj}:")
            lines.append(f"    both_succ={both_s}  {ni}_only={i_only}  "
                         f"{nj}_only={j_only}  both_fail={both_f}")
            if ks:
                lines.append(f"    top discriminating feature: {top_feat}  KS={top_ks:.3f}")
            pairwise_out[(ni, nj)] = {
                'both_success': both_s,
                f'{ni}_only': i_only,
                f'{nj}_only': j_only,
                'both_fail': both_f,
                'ks_discriminating': {k: v for k, v in sorted(
                    ks.items(), key=lambda x: -x[1]['ks_stat'])[:5]},
            }

    # ── decision trees per group ──────────────────────────────────────────────
    lines += ["", "Surrogate trees per group:"]
    trees_out = {}
    X_tree = np.where(np.isfinite(X), X, 0.0)
    for label, mask in [('universal_success', all_suc), ('universal_fail', all_fail)]:
        y = mask.astype(int)
        if len(np.unique(y)) < 2 or mask.sum() < 20:
            continue
        dt = DecisionTreeClassifier(max_depth=4, random_state=0)
        dt.fit(X_tree, y)
        acc = float(dt.score(X_tree, y))
        imp = dict(zip(feat_names, dt.feature_importances_.tolist()))
        lines.append(f"\n  {label} (n={mask.sum()}, tree_acc={acc:.3f}):")
        top5 = sorted(imp.items(), key=lambda x: -x[1])[:5]
        for k, v in top5:
            lines.append(f"    {k:<24}  imp={v:.4f}")
        trees_out[label] = {'accuracy': acc, 'importances': imp,
                            'text': export_text(dt, feature_names=feat_names)}

    out_dict = {
        'per_model': per_model_out,
        'universal_success_n': n_all_suc,
        'universal_success_rate': float(n_all_suc / N),
        'universal_fail_n': n_all_fail,
        'universal_fail_rate': float(n_all_fail / N),
        'group_features': group_features_out,
        'ks_suc_vs_fail': ks_suc_vs_fail,
        'unique_success': unique_out,
        'unique_fail': unique_fail_out,
        'pairwise': {f'{ni}_vs_{nj}': v for (ni, nj), v in pairwise_out.items()},
        'surrogate_trees': trees_out,
    }
    return "\n".join(lines), out_dict


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser("Edge-regime diagnosis for temporal link prediction")
    p.add_argument('--predictions',  required=True,
                   help='CSV from --save_test_predictions (edge_index, rank). '
                        'First file is the primary model; drives feature computation '
                        'and single-model analysis.')
    p.add_argument('--predictions2', default=None,
                   help='Second model CSV for legacy two-model comparison')
    p.add_argument('--compare', nargs='*', default=[],
                   help='Additional model CSVs for N-model comparison '
                        '(combined with --predictions2 if also given)')
    p.add_argument('--compare_names', nargs='*', default=[],
                   help='Model names for --compare files (same order). '
                        'Defaults to basename-derived names.')
    p.add_argument('--model_name',   default='model1')
    p.add_argument('--model2_name',  default='model2')
    p.add_argument('--dataset_name', required=True)
    p.add_argument('--rank_bins',    default='1,10,100',
                   help='Comma-separated rank thresholds, e.g. "1,10,100" → '
                        'rank=1 | 2-10 | 11-100 | >100')
    p.add_argument('--K',            type=int, default=10,
                   help='Binary success threshold for KS tests and tree (rank ≤ K)')
    p.add_argument('--hard_fail_rank', type=int, default=100,
                   help='Hard-failure threshold (rank > this is "hard fail")')
    p.add_argument('--out_dir',      default='analysis_results')
    p.add_argument('--eval_neg_strategy', default='historical',
                   choices=['random', 'historical', 'inductive'])
    p.add_argument('--use_test_history', action='store_true', default=False,
                   help='Streaming mode: update history with each test edge before the next. '
                        'Default: static train+val history only.')
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

    # derive run id from predictions filename, e.g.
    # "saved_results/pfx_link_tgbl-wiki_TPNet_seed0_test_predictions.csv"
    # → run_id = "pfx_link_tgbl-wiki_TPNet_seed0"
    pred_stem = os.path.splitext(os.path.basename(args.predictions))[0]
    run_id = pred_stem.replace('_test_predictions', '').replace('_val_predictions', '')
    out_dir = os.path.join(args.out_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    rank_bins = parse_rank_bins(args.rank_bins)
    hist_mode = "train+val+test-so-far" if args.use_test_history else "train+val only"

    # ── load dataset ──────────────────────────────────────────────────────────
    logger.info(f"Loading {args.dataset_name} ...")
    if args.dataset_name in CLASSIC_DATASETS:
        _, _, _, train_data, val_data, test_data, _, _ = \
            get_link_prediction_data_classic(dataset_name=args.dataset_name,
                                             eval_neg_strategy=args.eval_neg_strategy,
                                             logger=logger)
    else:
        _, _, _, train_data, val_data, test_data, _, _ = \
            get_link_prediction_data(dataset_name=args.dataset_name, logger=logger)

    # ── load predictions & sanity checks ─────────────────────────────────────
    pred_df      = pd.read_csv(args.predictions)
    edge_indices = pred_df['edge_index'].to_numpy()
    ranks        = pred_df['rank'].to_numpy()

    assert len(edge_indices) > 0, "Prediction CSV is empty"
    assert ranks.min() >= 1,      f"Ranks must be ≥ 1, got min={ranks.min()}"
    assert edge_indices.max() < test_data.num_interactions, \
        f"edge_index {edge_indices.max()} out of range (test_data size={test_data.num_interactions})"
    if len(edge_indices) != len(np.unique(edge_indices)):
        logger.warning("Duplicate edge_indices found in predictions CSV")

    n_test    = len(ranks)
    rr        = 1.0 / ranks.astype(float)
    bin_labels = assign_rank_bin(ranks, rank_bins)

    logger.info(f"{n_test} predictions | rank-1 rate: {(ranks==1).mean()*100:.1f}% "
                f"| overall MRR: {rr.mean():.4f}")

    # ── features ──────────────────────────────────────────────────────────────
    logger.info(f"Building history ({hist_mode}) ...")
    srcs_h, dsts_h, times_h = build_history(train_data, val_data)

    src_test  = test_data.src_node_ids[edge_indices]
    dst_test  = test_data.dst_node_ids[edge_indices]
    time_test = test_data.node_interact_times[edge_indices]

    logger.info("Computing features ...")
    features   = compute_features(src_test, dst_test, time_test,
                                  srcs_h, dsts_h, times_h,
                                  update_with_test_history=args.use_test_history)

    degen      = check_degenerate(features)
    # analysis features for KS / surrogate attribution: drop degenerate AND leaky
    # (future-derived) features; the latter stay in `features` for stratified reporting only.
    af         = {k: v for k, v in features.items()
                  if k not in degen and k not in LEAKY_FEATURES}

    # ── rank-bin summary ──────────────────────────────────────────────────────
    summary_df = rank_bin_summary(ranks, bin_labels, features)

    # ── KS tests ──────────────────────────────────────────────────────────────
    logger.info("KS tests ...")
    binary_labels = (ranks <= args.K).astype(np.int32)
    ks_res        = ks_tests(af, binary_labels)

    # hard-failure: rank=1 vs rank>hard_fail_rank
    hf_raw    = np.where(ranks == 1, 1, np.where(ranks > args.hard_fail_rank, 0, -1))
    hf_mask   = hf_raw >= 0
    ks_hf     = ks_tests({k: v[hf_mask] for k, v in af.items()}, hf_raw[hf_mask])

    # ── bucketed MRR ──────────────────────────────────────────────────────────
    logger.info("Bucketed MRR ...")
    bmrr_feats = ['pair_count', 'time_since_last', 'dst_popularity', 'src_popularity',
                  'src_recency', 'dst_recency', 'bipartite_cf_score',
                  'src_new', 'dst_new', 'both_new', 'struct_proxy']
    bmrr = {f: bucketed_mrr(features[f], ranks) for f in bmrr_feats if f in features}

    # ── decision tree ─────────────────────────────────────────────────────────
    logger.info("Fitting decision tree ...")
    tree_text, tree_imp, tree_metrics = tree_analysis(af, binary_labels)

    # ── two-model comparison (legacy) ────────────────────────────────────────
    comp_str  = ""
    comp_data = {}
    if args.predictions2:
        logger.info(f"Loading {args.predictions2} for comparison ...")
        p2_df  = pd.read_csv(args.predictions2)
        merged = pred_df.merge(p2_df, on='edge_index', suffixes=('_m1', '_m2'))
        ei2     = merged['edge_index'].to_numpy()
        r_m1    = merged['rank_m1'].to_numpy()
        r_m2    = merged['rank_m2'].to_numpy()
        rr_m1   = 1.0 / r_m1.astype(float)
        rr_m2   = 1.0 / r_m2.astype(float)
        delta   = rr_m2 - rr_m1   # positive = model2 better

        # map ei2 → positions in features array
        ei_to_pos = {int(e): i for i, e in enumerate(edge_indices)}
        pos2 = np.array([ei_to_pos[int(e)] for e in ei2])
        af2  = {k: v[pos2] for k, v in af.items()}

        m1r1 = r_m1 == 1;  m2r1 = r_m2 == 1
        m1hf = r_m1 > args.hard_fail_rank;  m2hf = r_m2 > args.hard_fail_rank
        groups = {
            'both_rank1':                 m1r1 & m2r1,
            f'{args.model_name}_only_r1': m1r1 & ~m2r1,
            f'{args.model2_name}_only_r1': ~m1r1 & m2r1,
            'both_hard_fail':             m1hf & m2hf,
        }
        comp_rows = []
        for gn, gm in groups.items():
            comp_rows.append({
                'group': gn, 'n': int(gm.sum()),
                '%': f"{100*gm.mean():.1f}",
                f'MRR_{args.model_name}':  f"{rr_m1[gm].mean():.4f}" if gm.any() else 'n/a',
                f'MRR_{args.model2_name}': f"{rr_m2[gm].mean():.4f}" if gm.any() else 'n/a',
            })
        comp_df = pd.DataFrame(comp_rows)

        # delta-RR KS: edges where model1 clearly better vs model2 clearly better
        dl = np.where(delta < -0.01, 1, np.where(delta > 0.01, 0, -1))
        dm = dl >= 0
        ks_delta = ks_tests({k: v[dm] for k, v in af2.items()}, dl[dm]) if dm.sum() > 10 else {}

        comp_str = (
            f"\n═══ MODEL COMPARISON: {args.model_name} vs {args.model2_name} ═══\n"
            f"{comp_df.to_string(index=False)}\n\n"
            f"ΔRR = RR({args.model2_name}) − RR({args.model_name})  "
            f"mean={delta.mean():.4f}  std={delta.std():.4f}  "
            f"[positive = {args.model2_name} better]\n"
        )
        if ks_delta:
            comp_str += (
                f"\nKS tests on features: {args.model_name}-better vs {args.model2_name}-better\n"
                + fmt_ks(ks_delta) + "\n"
            )
        comp_data = {
            'group_summary': comp_df.to_dict(orient='records'),
            'delta_rr_mean': float(delta.mean()), 'delta_rr_std': float(delta.std()),
            'ks_delta_rr':   ks_delta,
        }

    # ── N-model comparison ────────────────────────────────────────────────────
    multi_str  = ""
    multi_data = {}
    # Collect all extra model files: from --compare, and --predictions2 if not already covered
    extra_files = list(args.compare or [])
    extra_names = list(args.compare_names or [])
    if args.predictions2 and args.predictions2 not in extra_files:
        extra_files.insert(0, args.predictions2)
        extra_names.insert(0, args.model2_name)

    if extra_files:
        # auto-name extras that weren't named
        while len(extra_names) < len(extra_files):
            stem = os.path.splitext(os.path.basename(extra_files[len(extra_names)]))[0]
            extra_names.append(stem.replace('_test_predictions', ''))

        all_model_dfs = {args.model_name: pred_df}
        for path, name in zip(extra_files, extra_names):
            logger.info(f"Loading {path} for multi-model comparison ({name}) ...")
            all_model_dfs[name] = pd.read_csv(path)

        logger.info(f"Running {len(all_model_dfs)}-model intersection analysis ...")
        multi_str, multi_data = multi_model_comparison(
            all_model_dfs, af, edge_indices,
            K=args.K, hard_fail_rank=args.hard_fail_rank,
        )

        # write per-model rank columns into edge_diagnostics for easy downstream use
        for mname, mdf in all_model_dfs.items():
            ei_to_rk = {int(e): int(r) for e, r in
                        zip(mdf['edge_index'].to_numpy(), mdf['rank'].to_numpy())}
            diag_ranks = np.array([ei_to_rk.get(int(e), 0) for e in edge_indices])
            pred_df[f'rank_{mname}'] = np.where(diag_ranks > 0, diag_ranks, np.nan)

    # ── edge_diagnostics.csv ──────────────────────────────────────────────────
    logger.info("Writing edge_diagnostics.csv ...")
    diag = {'edge_index': edge_indices, 'src': src_test, 'dst': dst_test,
            'time': time_test, 'rank': ranks, 'rr': rr, 'rank_bin': bin_labels}
    diag.update(features)
    pd.DataFrame(diag).to_csv(os.path.join(out_dir, 'edge_diagnostics.csv'), index=False)

    # ── build report ──────────────────────────────────────────────────────────
    lines = [
        f"Edge-regime diagnosis — {args.dataset_name}",
        f"Prediction file : {args.predictions}",
        f"History mode    : {hist_mode}",
        f"K (binary)      : {args.K}   hard_fail_rank={args.hard_fail_rank}",
        f"Rank-1 rate     : {(ranks==1).mean()*100:.1f}%  ({(ranks==1).sum()}/{n_test})",
        f"Overall MRR     : {rr.mean():.4f}",
        f"Degenerate feats: {', '.join(degen) if degen else 'none'}",
        "",
        "═══ RANK-BIN SUMMARY ═══",
        summary_df.to_string(index=False),
        "",
        f"═══ KS TESTS: rank≤{args.K} (success) vs rank>{args.K} (fail) ═══",
        "  [time_since_last evaluated on repeated pairs only]",
        fmt_ks(ks_res),
        "",
        f"═══ KS TESTS: rank=1 vs rank>{args.hard_fail_rank} (hard failure) ═══",
        fmt_ks(ks_hf),
        "",
        "═══ MRR BY FEATURE QUANTILE ═══",
        fmt_bmrr(bmrr),
        "",
        f"═══ DECISION TREE (K={args.K}, max_depth=4) ═══",
        "  Feature importances:",
    ]
    for feat, imp in sorted(tree_imp.items(), key=lambda x: -x[1]):
        lines.append(f"    {feat:<24} {imp:.4f}")
    lines += [
        "  Metrics: " + "  |  ".join(f"{k}={v:.4f}" for k, v in tree_metrics.items()),
        "",
        tree_text,
        comp_str,
        multi_str,
    ]

    report = "\n".join(lines)
    print(report)
    with open(os.path.join(out_dir, 'report.txt'), 'w') as f:
        f.write(report)

    # ── results.json ──────────────────────────────────────────────────────────
    out = {
        'config': vars(args),
        'history_mode': hist_mode,
        'n_test_edges': n_test,
        'overall_mrr':  float(rr.mean()),
        'rank1_rate':   float((ranks == 1).mean()),
        'degenerate_features': list(degen),
        'rank_bin_summary':    summary_df.to_dict(orient='records'),
        'ks_success_vs_fail':  ks_res,
        'ks_rank1_vs_hard_fail': ks_hf,
        'bucketed_mrr':        {k: v for k, v in bmrr.items() if v},
        'tree_importances':    tree_imp,
        'tree_metrics':        tree_metrics,
    }
    if comp_data:
        out['comparison'] = comp_data
    if multi_data:
        out['multi_model'] = multi_data
        # also write a standalone multi-model report
        multi_report_path = os.path.join(out_dir, 'multi_model_report.txt')
        with open(multi_report_path, 'w') as f:
            f.write(multi_str)
        multi_json_path = os.path.join(out_dir, 'multi_model_results.json')
        with open(multi_json_path, 'w') as f:
            json.dump(multi_data, f, indent=2, default=str)
        logger.info(f"Multi-model results: {multi_report_path} | {multi_json_path}")
    with open(os.path.join(out_dir, 'results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)

    logger.info(f"Done. Outputs in {out_dir}/  "
                f"(report.txt | results.json | edge_diagnostics.csv)")


if __name__ == '__main__':
    main()
