"""
Fusion — a findings-driven multi-block temporal link predictor.

Motivation (from the per-edge diagnostic):
  * the architectural families are complementary on novel edges (oracle headroom);
  * but WHICH model wins is not recoverable from edge features (a feature router
    fails) -> the gating must be INTERNAL, learned over the blocks' own
    representations.

So Fusion carries the complementary building blocks in one network and fuses
them with a learned per-edge softmax gate (mixture-of-experts over blocks):
  - 'walk'   : TPNet's random-feature temporal-walk projections (-> novel/structural edges)
  - 'cooc'   : DyGFormer's neighbour co-occurrence transformer (-> shared-neighbour edges)
  - 'memory' : TGN's recurrent node memory                       (-> repeated/recurrence edges)

Each block reuses the original, tested implementation (we only add the gate).
Blocks are selectable via `blocks=(...)` so we can validate walk+cooc first
(single stateful component: the random projections) before adding memory
(which also needs the memory-bank backup/reload machinery in the train loop).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.TPNet import TPNet
from models.DyGFormer import DyGFormer
from models.MemoryModel import MemoryModel


class Fusion(nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler,
                 time_feat_dim: int, output_dim: int, dropout: float, device: str,
                 blocks=('walk', 'cooc'),
                 # walk (TPNet) block
                 random_projections=None, num_neighbors: int = 20, num_layers: int = 2,
                 time_encoder_type: str = 'sinusoidal',
                 # cooc (DyGFormer) block
                 channel_embedding_dim: int = 50, patch_size: int = 1,
                 max_input_sequence_length: int = 32, num_heads: int = 2,
                 # memory (TGN) block — time-shift stats required only if 'memory' in blocks
                 src_node_mean_time_shift: float = 0.0, src_node_std_time_shift: float = 1.0,
                 dst_node_mean_time_shift_dst: float = 0.0, dst_node_std_time_shift: float = 1.0,
                 # gate
                 gate_hidden_dim: int = 64, gate_conf: bool = False):
        super(Fusion, self).__init__()
        assert len(blocks) >= 1
        self.blocks = tuple(blocks)
        self.output_dim = output_dim
        self.num_neighbors = num_neighbors
        self.device = device

        self.sub = nn.ModuleDict()
        if 'walk' in self.blocks:
            assert random_projections is not None, "'walk' block needs a RandomProjectionModule"
            self.sub['walk'] = TPNet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                     neighbor_sampler=neighbor_sampler, time_feat_dim=time_feat_dim,
                                     output_dim=output_dim, dropout=dropout, random_projections=random_projections,
                                     num_layers=num_layers, num_neighbors=num_neighbors, device=device,
                                     time_encoder_type=time_encoder_type)
        if 'cooc' in self.blocks:
            self.sub['cooc'] = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                         neighbor_sampler=neighbor_sampler, time_feat_dim=time_feat_dim,
                                         output_dim=output_dim, channel_embedding_dim=channel_embedding_dim,
                                         patch_size=patch_size, num_layers=num_layers, num_heads=num_heads,
                                         dropout=dropout, max_input_sequence_length=max_input_sequence_length,
                                         device=device, use_cooccurrence=True)
        if 'memory' in self.blocks:
            self.sub['memory'] = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                             neighbor_sampler=neighbor_sampler, output_dim=output_dim,
                                             time_feat_dim=time_feat_dim, model_name='TGN', num_layers=1,
                                             num_heads=num_heads, dropout=dropout,
                                             src_node_mean_time_shift=src_node_mean_time_shift,
                                             src_node_std_time_shift=src_node_std_time_shift,
                                             dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                             dst_node_std_time_shift=dst_node_std_time_shift, device=device)

        # lightweight trainable per-block adapter (normalises each frozen block's
        # embedding into a comparable space before gating)
        self.block_proj = nn.ModuleDict({b: nn.Linear(output_dim, output_dim) for b in self.blocks})

        # learned per-edge gate over blocks. Input = each block's [src;dst] embedding,
        # optionally augmented with each block's per-edge compatibility (dot + cosine of
        # src,dst in that block's own space) — an expert-confidence / edge-specific signal.
        n_blocks = len(self.blocks)
        self.gate_conf = gate_conf
        gate_in_dim = 2 * output_dim * n_blocks + (2 * n_blocks if gate_conf else 0)
        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, gate_hidden_dim), nn.ReLU(),
            nn.Linear(gate_hidden_dim, n_blocks))
        self._last_gate = None  # store last gate weights for analysis

        # expose state for the training loop (so it can drive memory / RP exactly as for TGN / TPNet)
        self.random_projections = random_projections
        if 'memory' in self.blocks:
            self.memory_bank = self.sub['memory'].memory_bank

    def _block_embeddings(self, src_node_ids, dst_node_ids, node_interact_times,
                          edge_ids, edges_are_positive, num_neighbors):
        srcs, dsts = [], []
        for b in self.blocks:
            if b == 'walk':
                s, d = self.sub['walk'].compute_src_dst_node_temporal_embeddings(
                    src_node_ids, dst_node_ids, node_interact_times)
            elif b == 'cooc':
                s, d = self.sub['cooc'].compute_src_dst_node_temporal_embeddings(
                    src_node_ids, dst_node_ids, node_interact_times)
            elif b == 'memory':
                s, d = self.sub['memory'].compute_src_dst_node_temporal_embeddings(
                    src_node_ids=src_node_ids, dst_node_ids=dst_node_ids,
                    node_interact_times=node_interact_times, edge_ids=edge_ids,
                    edges_are_positive=edges_are_positive, num_neighbors=num_neighbors)
            srcs.append(s)
            dsts.append(d)
        return srcs, dsts

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids, dst_node_ids, node_interact_times,
                                                 edge_ids=None, edges_are_positive: bool = False,
                                                 num_neighbors: int = 20, time_gap: int = 2000):
        srcs, dsts = self._block_embeddings(src_node_ids, dst_node_ids, node_interact_times,
                                            edge_ids, edges_are_positive, num_neighbors)
        # per-expert edge-compatibility (computed in each block's OWN space, before the adapter):
        # dot product and cosine of (src, dst) = how strongly block b links this pair (confidence).
        conf = None
        if self.gate_conf and len(self.blocks) > 1:
            dots = [(s * d).sum(dim=-1, keepdim=True) for s, d in zip(srcs, dsts)]
            coss = [F.cosine_similarity(s, d, dim=-1).unsqueeze(-1) for s, d in zip(srcs, dsts)]
            conf = torch.cat(dots + coss, dim=1)  # (batch, 2*n_blocks)
        # trainable per-block adapter
        srcs = [self.block_proj[b](s) for b, s in zip(self.blocks, srcs)]
        dsts = [self.block_proj[b](d) for b, d in zip(self.blocks, dsts)]
        if len(self.blocks) == 1:
            return srcs[0], dsts[0]
        # (batch, n_blocks, output_dim)
        src_stack = torch.stack(srcs, dim=1)
        dst_stack = torch.stack(dsts, dim=1)
        # per-edge gate from the blocks' own representations (+ optional confidence signal)
        gate_parts = [src_stack.flatten(start_dim=1), dst_stack.flatten(start_dim=1)]
        if conf is not None:
            gate_parts.append(conf)
        gate_in = torch.cat(gate_parts, dim=1)
        weights = torch.softmax(self.gate(gate_in), dim=1)          # (batch, n_blocks)
        self._last_gate = weights.detach()
        w = weights.unsqueeze(-1)                                   # (batch, n_blocks, 1)
        src_emb = (src_stack * w).sum(dim=1)                        # (batch, output_dim)
        dst_emb = (dst_stack * w).sum(dim=1)
        return src_emb, dst_emb

    def set_neighbor_sampler(self, neighbor_sampler):
        for b in self.blocks:
            self.sub[b].set_neighbor_sampler(neighbor_sampler)

    def load_pretrained(self, ckpt_paths: dict, freeze: bool = True, logger=None):
        """Warm-start each block from its donor single-model checkpoint and (optionally) freeze it.
        ckpt_paths: {block_name: path}. The checkpoint stores nn.Sequential(backbone, decoder)
        under 'model'; we load the backbone ('0.'-prefixed) params into the block (strict=False,
        so the random-projection buffers / decoder are tolerated)."""
        import torch
        for b in self.blocks:
            path = ckpt_paths.get(b)
            if path is None:
                continue
            sd = torch.load(path, map_location='cpu')['model']
            backbone_sd = {k[len('0.'):]: v for k, v in sd.items() if k.startswith('0.')}
            missing, unexpected = self.sub[b].load_state_dict(backbone_sd, strict=False)
            if logger is not None:
                logger.info(f"[Fusion] warm-started '{b}' from {path} "
                            f"(missing={len(missing)}, unexpected={len(unexpected)})")
            if freeze:
                for p in self.sub[b].parameters():
                    p.requires_grad = False
        # only block_proj + gate (and, if not frozen, the blocks) remain trainable
        self._frozen = freeze
