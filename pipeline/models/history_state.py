import numpy as np


class HistoryState:
    """
    Chronological edge-history for explicit feature computation.

    Feature layout returned by compute_features() (indices 0-10 always present):
      0  log1p(pair_count)     1  pair_seen          2  log1p(pair_gap)
      3  log1p(src_count)      4  log1p(dst_count)
      5  log1p(src_gap)        6  log1p(dst_gap)
      7  log1p(node_count_u)   8  log1p(node_count_v)
      9  log1p(node_gap_u)     10 log1p(node_gap_v)
    if use_cf:
      11 log1p(cf_support)
    if use_feat_sim and use_edge_feats and edge_feat_dim > 0:
      +0 sim_src  +1 sim_dst  +2 1-sim_src  +3 1-sim_dst

    All gap / count values use large_gap_value when the entity has not been seen.
    """

    def __init__(self, large_gap_value: float = 1e9, max_cf_neighbors: int = 30,
                 max_cf_sources: int = 30, use_edge_feats: bool = False,
                 edge_feat_dim: int = 0):
        self.large_gap_value = large_gap_value
        self.max_cf_neighbors = max_cf_neighbors
        self.max_cf_sources = max_cf_sources
        self.use_edge_feats = use_edge_feats
        self.edge_feat_dim = edge_feat_dim
        self._init_dicts()

    def _init_dicts(self):
        self.pair_count = {}
        self.pair_last_time = {}
        self.src_count = {}
        self.dst_count = {}
        self.node_count = {}
        self.src_last_time = {}
        self.dst_last_time = {}
        self.node_last_time = {}
        self.out_neighbors = {}
        self.in_neighbors = {}
        if self.use_edge_feats:
            self.edge_feat_sum_src = {}
            self.edge_feat_sum_dst = {}
            self.edge_feat_n_src = {}
            self.edge_feat_n_dst = {}

    def reset(self):
        self._init_dicts()

    def update(self, src_ids, dst_ids, times, edge_feats=None):
        """Update with a batch of observed positive edges (chronological order assumed)."""
        for i in range(len(src_ids)):
            u, v, t = int(src_ids[i]), int(dst_ids[i]), float(times[i])
            pair = (u, v)
            self.pair_count[pair] = self.pair_count.get(pair, 0) + 1
            self.pair_last_time[pair] = t
            self.src_count[u] = self.src_count.get(u, 0) + 1
            self.dst_count[v] = self.dst_count.get(v, 0) + 1
            self.node_count[u] = self.node_count.get(u, 0) + 1
            self.node_count[v] = self.node_count.get(v, 0) + 1
            self.src_last_time[u] = t
            self.dst_last_time[v] = t
            self.node_last_time[u] = t
            self.node_last_time[v] = t
            if u not in self.out_neighbors:
                self.out_neighbors[u] = set()
            self.out_neighbors[u].add(v)
            if v not in self.in_neighbors:
                self.in_neighbors[v] = set()
            self.in_neighbors[v].add(u)
            if self.use_edge_feats and edge_feats is not None:
                feat = np.asarray(edge_feats[i], dtype=np.float64)
                if u not in self.edge_feat_sum_src:
                    self.edge_feat_sum_src[u] = feat.copy()
                    self.edge_feat_n_src[u] = 1
                else:
                    self.edge_feat_sum_src[u] += feat
                    self.edge_feat_n_src[u] += 1
                if v not in self.edge_feat_sum_dst:
                    self.edge_feat_sum_dst[v] = feat.copy()
                    self.edge_feat_n_dst[v] = 1
                else:
                    self.edge_feat_sum_dst[v] += feat
                    self.edge_feat_n_dst[v] += 1

    def feature_dim(self, use_cf: bool = True, use_feat_sim: bool = False) -> int:
        dim = 11
        if use_cf:
            dim += 1
        if use_feat_sim and self.use_edge_feats and self.edge_feat_dim > 0:
            dim += 4
        return dim

    def compute_features(self, src_ids, dst_ids, times, edge_feats=None,
                          use_cf: bool = True, use_feat_sim: bool = False) -> np.ndarray:
        """
        Compute log1p-scaled feature vectors from history state *before* any update.
        Call update() after scoring to maintain the no-leakage invariant.

        Returns float32 array of shape (N, feature_dim).
        When use_feat_sim=True but edge_feats=None (e.g. negative candidates),
        the similarity features are filled with 0 so the output dim is unchanged.
        """
        N = len(src_ids)
        lgv = self.large_gap_value
        feat_dim = self.feature_dim(use_cf, use_feat_sim)

        # precompute CF reach-count per unique source (batched for efficiency)
        reach_cache = {}
        if use_cf:
            for u in set(int(s) for s in src_ids):
                reach_cache[u] = self._compute_reach_count(u)

        rows = np.zeros((N, feat_dim), dtype=np.float32)
        for i in range(N):
            u, v, t = int(src_ids[i]), int(dst_ids[i]), float(times[i])
            pair = (u, v)
            pc = self.pair_count.get(pair, 0)
            pair_gap = (t - self.pair_last_time[pair]) if pc > 0 else lgv
            sc = self.src_count.get(u, 0)
            dc = self.dst_count.get(v, 0)
            src_gap = (t - self.src_last_time[u]) if u in self.src_last_time else lgv
            dst_gap = (t - self.dst_last_time[v]) if v in self.dst_last_time else lgv
            nc_u = self.node_count.get(u, 0)
            nc_v = self.node_count.get(v, 0)
            ng_u = (t - self.node_last_time[u]) if u in self.node_last_time else lgv
            ng_v = (t - self.node_last_time[v]) if v in self.node_last_time else lgv

            rows[i, 0] = np.log1p(pc)
            rows[i, 1] = 1.0 if pc > 0 else 0.0
            rows[i, 2] = np.log1p(pair_gap)
            rows[i, 3] = np.log1p(sc)
            rows[i, 4] = np.log1p(dc)
            rows[i, 5] = np.log1p(src_gap)
            rows[i, 6] = np.log1p(dst_gap)
            rows[i, 7] = np.log1p(nc_u)
            rows[i, 8] = np.log1p(nc_v)
            rows[i, 9] = np.log1p(ng_u)
            rows[i, 10] = np.log1p(ng_v)

            col = 11
            if use_cf:
                rows[i, col] = np.log1p(reach_cache[u].get(v, 0))
                col += 1

            if use_feat_sim and self.use_edge_feats and self.edge_feat_dim > 0:
                if edge_feats is not None:
                    ef = np.asarray(edge_feats[i], dtype=np.float32)
                    s_src = self._cosine_sim_mean(
                        ef, self.edge_feat_sum_src.get(u), self.edge_feat_n_src.get(u, 0))
                    s_dst = self._cosine_sim_mean(
                        ef, self.edge_feat_sum_dst.get(v), self.edge_feat_n_dst.get(v, 0))
                else:
                    s_src, s_dst = 0.0, 0.0
                rows[i, col:col + 4] = [s_src, s_dst, 1.0 - s_src, 1.0 - s_dst]

        assert np.isfinite(rows).all(), "HistoryState.compute_features produced non-finite values"
        return rows

    def _compute_reach_count(self, u: int) -> dict:
        """
        Bipartite CF: v -> number of co-visitor sources of u that have interacted with v.
        Caps at max_cf_neighbors / max_cf_sources for performance.
        Uses sorted() for deterministic output regardless of set insertion order.
        """
        out_u = self.out_neighbors.get(u)
        if not out_u:
            return {}
        co_sources = set()
        for x in sorted(out_u)[:self.max_cf_neighbors]:
            in_x = self.in_neighbors.get(x)
            if in_x:
                for u2 in sorted(in_x)[:self.max_cf_sources]:
                    co_sources.add(u2)
        reach = {}
        for u2 in co_sources:
            out_u2 = self.out_neighbors.get(u2)
            if out_u2:
                for v2 in sorted(out_u2)[:self.max_cf_neighbors]:
                    reach[v2] = reach.get(v2, 0) + 1
        return reach

    def _cosine_sim_mean(self, feat: np.ndarray, feat_sum, n: int) -> float:
        if n == 0 or feat_sum is None:
            return 0.0
        mean_feat = feat_sum / n
        nf = np.linalg.norm(feat)
        nm = np.linalg.norm(mean_feat)
        if nf < 1e-10 or nm < 1e-10:
            return 0.0
        return float(np.dot(feat, mean_feat) / (nf * nm))

    def backup(self) -> dict:
        """Snapshot all state dicts for backup/restore across train-val-test boundaries."""
        bk = {
            'pair_count': dict(self.pair_count),
            'pair_last_time': dict(self.pair_last_time),
            'src_count': dict(self.src_count),
            'dst_count': dict(self.dst_count),
            'node_count': dict(self.node_count),
            'src_last_time': dict(self.src_last_time),
            'dst_last_time': dict(self.dst_last_time),
            'node_last_time': dict(self.node_last_time),
            'out_neighbors': {k: set(v) for k, v in self.out_neighbors.items()},
            'in_neighbors': {k: set(v) for k, v in self.in_neighbors.items()},
        }
        if self.use_edge_feats:
            bk['edge_feat_sum_src'] = {
                k: v.copy() for k, v in getattr(self, 'edge_feat_sum_src', {}).items()}
            bk['edge_feat_sum_dst'] = {
                k: v.copy() for k, v in getattr(self, 'edge_feat_sum_dst', {}).items()}
            bk['edge_feat_n_src'] = dict(getattr(self, 'edge_feat_n_src', {}))
            bk['edge_feat_n_dst'] = dict(getattr(self, 'edge_feat_n_dst', {}))
        return bk

    def restore(self, bk: dict):
        """Restore state from a backup produced by backup()."""
        self.pair_count = dict(bk['pair_count'])
        self.pair_last_time = dict(bk['pair_last_time'])
        self.src_count = dict(bk['src_count'])
        self.dst_count = dict(bk['dst_count'])
        self.node_count = dict(bk['node_count'])
        self.src_last_time = dict(bk['src_last_time'])
        self.dst_last_time = dict(bk['dst_last_time'])
        self.node_last_time = dict(bk['node_last_time'])
        self.out_neighbors = {k: set(v) for k, v in bk['out_neighbors'].items()}
        self.in_neighbors = {k: set(v) for k, v in bk['in_neighbors'].items()}
        if self.use_edge_feats and 'edge_feat_sum_src' in bk:
            self.edge_feat_sum_src = {k: v.copy() for k, v in bk['edge_feat_sum_src'].items()}
            self.edge_feat_sum_dst = {k: v.copy() for k, v in bk['edge_feat_sum_dst'].items()}
            self.edge_feat_n_src = dict(bk['edge_feat_n_src'])
            self.edge_feat_n_dst = dict(bk['edge_feat_n_dst'])
