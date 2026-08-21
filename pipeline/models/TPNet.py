import torch
import numpy as np
import torch.nn as nn
from utils.utils import NeighborSampler
import math
from models.modules import TimeEncoder, LinearTimeEncoder


class RandomProjectionModule(nn.Module):
    def __init__(self, node_num: int, edge_num: int, dim_factor: int, num_layer: int,
                 device: str, use_matrix: bool, beginning_time: np.float64, not_scale: bool,
                 enforce_dim: int, adaptive_lambda: bool = True, time_decay_weight: float = 1e-6,
                 lambda_context: bool = True, target_bound_mode: str = 'sqrt_Nd',
                 target_bound_fixed: float = 1000.0, norm_mode: str = 'frobenius',
                 lambda_alpha: float = 0.01, no_global_decay: bool = False,
                 control_mode: str = 'controller'):
        """
        :param adaptive_lambda: if True, lambda_val adapts during training/eval via _adapt_lambda().
                                if False, lambda_val stays fixed at time_decay_weight (original TPNet behaviour).
        :param time_decay_weight: initial (and fixed, when adaptive_lambda=False) value of the time decay weight.
        :param lambda_context: if True, lambda_val is appended as an extra input to the pairwise MLP.
                               Requires adaptive_lambda=True to be meaningful. Can be disabled for ablation studies.
        :param target_bound_mode: how to compute the target bound for _adapt_lambda.
                                  'fixed'      -- constant value given by target_bound_fixed.
                                  'sqrt_Nd'    -- sqrt(N * d), N = total node count (static).
                                  'sqrt_N_t_d' -- sqrt(N(t) * d), N(t) = unique nodes seen so far (grows over time).
        :param target_bound_fixed: constant target bound used when target_bound_mode == 'fixed'.
        :param norm_mode: which norm to use when measuring matrix size in _adapt_lambda.
                          'frobenius'  -- sqrt(Σ a_ij²), RMS of all entries (default).
                          'avg_row'    -- mean per-node row norm: (1/N) Σ_i ||row_i||_2.
                          'max_row'    -- maximum per-node row norm: max_i ||row_i||_2.
        :param lambda_alpha: feedback strength for lambda adaptation. Controls how fast lambda
                             reacts to the error ratio: lambda *= (error_ratio ** alpha).
                             Smaller values mean slower, smoother adaptation.
        :param control_mode: which norm-control strategy to use when adaptive_lambda=True.
                             'controller'  -- indirect: adjust lambda to drive norm toward bound (_adapt_lambda).
                             'projection'  -- direct: rescale matrices onto the norm ball when norm exceeds bound.
        """
        super(RandomProjectionModule, self).__init__()
        self.node_num = node_num
        self.edge_num = edge_num
        if enforce_dim != -1:
            self.dim = enforce_dim
        else:
            self.dim = min(int(math.log(self.edge_num * 2))
                           * dim_factor, node_num)
        self.num_layer = num_layer
        self.adaptive_lambda = adaptive_lambda
        self.lambda_context = lambda_context
        self.target_bound_mode = target_bound_mode
        self.target_bound_fixed = target_bound_fixed
        self.norm_mode = norm_mode
        self.lambda_alpha = lambda_alpha
        self.no_global_decay = no_global_decay
        self.control_mode = control_mode

        self.register_buffer('lambda_val', torch.tensor(time_decay_weight, device=device))

        # used only when target_bound_mode == 'sqrt_N_t_d': tracks which nodes have been seen
        if target_bound_mode == 'sqrt_N_t_d':
            self.register_buffer('seen_mask', torch.zeros(node_num, dtype=torch.bool, device=device))
        self.norm_history = []
        self.lambda_history = []
        self.target_bound_history = []
        self.rho_history = []

        self.begging_time = nn.Parameter(
            torch.tensor(beginning_time), requires_grad=False)
        self.now_time = nn.Parameter(torch.tensor(
            beginning_time), requires_grad=False)
        self.device = device
        self.random_projections = nn.ParameterList()
        self.use_matrix = use_matrix
        
        # RESTORED: Kept exactly as in your original snippet
        self.node_feature_dim = 128
        self.not_scale = not_scale
        
        # RESTORED: Exact formatting and logic of your original matrix block
        if self.use_matrix:
            self.dim = self.node_num
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(torch.eye(self.node_num), requires_grad=False))
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False))
        # otherwise, store the random projection of the temporal walk matrices
        else:
            for i in range(self.num_layer + 1):
                if i == 0:
                    self.random_projections.append(
                        nn.Parameter(torch.normal(0, 1 / math.sqrt(self.dim), (self.node_num, self.dim)),
                                     requires_grad=False))
                else:
                    self.random_projections.append(
                        nn.Parameter(torch.zeros_like(self.random_projections[i - 1]), requires_grad=False))
        
        self.raw_pairwise_dim = (2 * self.num_layer + 2) ** 2
        # lambda_val is appended as extra context to the MLP input only when both flags are set
        self.pair_wise_feature_dim = self.raw_pairwise_dim + (1 if (adaptive_lambda and lambda_context) else 0)

        self.mlp = nn.Sequential(nn.Linear(self.pair_wise_feature_dim, self.pair_wise_feature_dim * 4), nn.ReLU(),
                                 nn.Linear(self.pair_wise_feature_dim * 4, self.pair_wise_feature_dim))



    @property
    def target_bound(self) -> float:
        if self.target_bound_mode == 'fixed':
            return self.target_bound_fixed
        _, cols = self.random_projections[1].shape
        if self.target_bound_mode == 'sqrt_Nd':
            rows, _ = self.random_projections[1].shape
            return math.sqrt(rows * cols)
        # sqrt_N_t_d: use number of unique nodes seen so far
        n_seen = max(self.seen_mask.sum().item(), 1)
        return math.sqrt(n_seen * cols)

    def _compute_norm(self) -> list:
        norms = []
        for i in range(1, self.num_layer + 1):
            A = self.random_projections[i]
            if self.norm_mode == 'frobenius':
                norms.append(torch.norm(A, p='fro').item())
            elif self.norm_mode == 'avg_row':
                norms.append(torch.norm(A, dim=1).mean().item())
            elif self.norm_mode == 'max_row':
                norms.append(torch.norm(A, dim=1).max().item())
        return norms

    def _adapt_lambda(self):
        with torch.no_grad():
            norms = self._compute_norm()
            avg_norm = sum(norms) / len(norms)

            if avg_norm == 0:
                return

            error_ratio = avg_norm / self.target_bound
            self.lambda_val *= (error_ratio ** self.lambda_alpha)
            # Clamp schützt vor dem kompletten Kollaps (Overflow/Underflow)
            self.lambda_val.clamp_(1e-12, 100)

            self.norm_history.append(norms)
            self.lambda_history.append(self.lambda_val.item())
            self.target_bound_history.append(self.target_bound)

    def _project_to_bound(self, eps: float = 1e-12):
        with torch.no_grad():
            norms = self._compute_norm()
            bound = self.target_bound
            rhos = []
            for i, layer_norm in enumerate(norms, start=1):
                if layer_norm > bound:
                    rho = bound / (layer_norm + eps)
                    self.random_projections[i].data *= rho
                else:
                    rho = 1.0
                rhos.append(rho)
            norms = self._compute_norm()
            self.norm_history.append(norms)
            self.rho_history.append(rhos)
            self.lambda_history.append(self.lambda_val.item())
            self.target_bound_history.append(bound)

    def update(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        src_node_ids = torch.from_numpy(src_node_ids).to(self.device)
        dst_node_ids = torch.from_numpy(dst_node_ids).to(self.device)
        next_time = node_interact_times[-1]
        node_interact_times_t = torch.from_numpy(node_interact_times).to(dtype=torch.float, device=self.device)
        
        projection_mode = self.control_mode == "projection"
        if projection_mode:
            # Capacity-projected memory: no lambda-based decay inside RP memory
            time_weight = torch.ones((len(node_interact_times), 1), device=self.device)
        else:
            time_weight = torch.exp(-self.lambda_val * (next_time - node_interact_times_t))[:, None]

        delta_t = next_time - self.now_time.item()

        if not projection_mode:
            for i in range(1, self.num_layer + 1):
                scale = 1 if self.no_global_decay else delta_t
                decay_factor = torch.exp(-self.lambda_val * scale * i)
                self.random_projections[i].data *= decay_factor

        for i in range(self.num_layer, 0, -1):
            src_update_messages = self.random_projections[i - 1][dst_node_ids] * time_weight
            dst_update_messages = self.random_projections[i - 1][src_node_ids] * time_weight
            self.random_projections[i].scatter_add_(dim=0, index=src_node_ids[:, None].expand(-1, self.dim),
                                                    src=src_update_messages)
            self.random_projections[i].scatter_add_(dim=0, index=dst_node_ids[:, None].expand(-1, self.dim),
                                                    src=dst_update_messages)

        self.now_time.data = torch.tensor(next_time, device=self.device)

        if self.target_bound_mode == 'sqrt_N_t_d':
            self.seen_mask[src_node_ids] = True
            self.seen_mask[dst_node_ids] = True

        if self.adaptive_lambda:
            if self.control_mode == 'projection':
                self._project_to_bound()
            else:
                self._adapt_lambda()
        else:
            norms = self._compute_norm()
            if any(n > 0 for n in norms):
                self.norm_history.append(norms)
                self.lambda_history.append(self.lambda_val.item())
                self.target_bound_history.append(self.target_bound)

    def get_random_projections(self, node_ids: np.ndarray):
        """
        get the random projections of the give node ids.
        :param node_ids: np.ndarray, shape (batch,)
        :return:
        """
        random_projections = []
        for i in range(self.num_layer + 1):
            random_projections.append(self.random_projections[i][node_ids])
        return random_projections

    def get_pair_wise_feature(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray):
        src_random_projections = torch.stack(self.get_random_projections(src_node_ids), dim=1)
        dst_random_projections = torch.stack(self.get_random_projections(dst_node_ids), dim=1)
        random_projections = torch.cat([src_random_projections, dst_random_projections], dim=1)
        
        random_feature = torch.matmul(random_projections, random_projections.transpose(1, 2)).reshape(len(src_node_ids), -1)
        
        if not self.not_scale:
            random_feature = torch.log(torch.relu(random_feature) + 1.0)

        if self.adaptive_lambda and self.lambda_context:
            l_context = self.lambda_val.expand(random_feature.shape[0], 1)
            random_feature = torch.cat([random_feature, l_context], dim=1)

        return self.mlp(random_feature)

    def save_history(self, path: str):
        kwargs = dict(norm=np.array(self.norm_history), lambda_val=np.array(self.lambda_history),
                      target_bound=np.array(self.target_bound_history))
        if self.rho_history:
            kwargs['rho'] = np.array(self.rho_history)
        np.savez(path, **kwargs)

    def reset_random_projections(self, reset_zero=True):
        """
        reset the random projections
        """
        for i in range(1, self.num_layer + 1):
            nn.init.zeros_(self.random_projections[i])
        self.now_time.data = self.begging_time.clone()
        if not self.use_matrix and reset_zero:
            nn.init.normal_(
                self.random_projections[0], mean=0, std=1 / math.sqrt(self.dim))
        if self.target_bound_mode == 'sqrt_N_t_d':
            self.seen_mask.zero_()

    def backup_random_projections(self):
        """
        backup the random projections.
        :return: tuple of (now_time, random_projections, lambda_val, seen_mask or None)
        """
        return (
            self.now_time.clone(),
            [self.random_projections[i].clone() for i in range(1, self.num_layer + 1)],
            self.lambda_val.clone(),
            self.seen_mask.clone() if self.target_bound_mode == 'sqrt_N_t_d' else None,
        )

    def reload_random_projections(self, backup):
        """
        reload the random projections.
        lambda_val is restored so every val/test run starts from the training lambda.
        :param backup: tuple of (now_time, random_projections, lambda_val, seen_mask or None)
        """
        now_time, rp_list, lambda_val, seen_mask = backup
        self.now_time.data = now_time.clone()
        for i in range(1, self.num_layer + 1):
            self.random_projections[i].data = rp_list[i - 1].clone()
        self.lambda_val.data = lambda_val.clone()
        if seen_mask is not None:
            self.seen_mask.data = seen_mask.clone()


class TPNet(torch.nn.Module):
    def __init__(self, node_raw_features: np.ndarray, edge_raw_features: np.ndarray, neighbor_sampler: NeighborSampler,
                 time_feat_dim: int, output_dim: int, dropout: float, random_projections: RandomProjectionModule,
                 num_layers: int, num_neighbors: int, device: str, not_embedding=False,
                 time_encoder_type: str = 'sinusoidal'):
        """
        Time decay matrix Projection-based graph neural Network for temporal link prediction, named TPNet for short.
        :param node_raw_features: ndarray, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: ndarray, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: neighbor sampler
        :param time_feat_dim: int, dimension of time features (encodings)
        :param dropout: float, dropout rate
        :param random_projections: RandomProjectionModule, the projected time decay temporal walk matrices
        :param num_layers: int, number of embedding layers
        :param num_neighbors: int, number of sampled neighbors
        :param device: str, device
        """
        super(TPNet, self).__init__()

        self.node_raw_features = torch.from_numpy(
            node_raw_features.astype(np.float32)).to(device)
        self.edge_raw_features = torch.from_numpy(
            edge_raw_features.astype(np.float32)).to(device)

        self.node_feat_dim = self.node_raw_features.shape[1]
        self.edge_feat_dim = self.edge_raw_features.shape[1]
        self.time_feat_dim = time_feat_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.device = device
        self.not_embedding = not_embedding

        # number of nodes, including the padded node
        self.num_nodes = self.node_raw_features.shape[0]

        self.random_projections = random_projections
        if time_encoder_type == 'linear':
            self.time_encoder = LinearTimeEncoder(time_dim=time_feat_dim)
        else:
            self.time_encoder = TimeEncoder(time_dim=time_feat_dim)

        # embedding module
        if self.not_embedding:
            self.embedding_module = None
        else:
            self.embedding_module = TPNetEmbedding(node_raw_features=self.node_raw_features,
                                                edge_raw_features=self.edge_raw_features,
                                                neighbor_sampler=neighbor_sampler,
                                                time_encoder=self.time_encoder,
                                                node_feat_dim=self.node_feat_dim,
                                                edge_feat_dim=self.edge_feat_dim,
                                                time_feat_dim=self.time_feat_dim,
                                                output_dim=self.output_dim,
                                                num_layers=num_layers,
                                                num_neighbors=num_neighbors,
                                                dropout=self.dropout,
                                                random_projections=self.random_projections)

    def compute_src_dst_node_temporal_embeddings(self, src_node_ids: np.ndarray, dst_node_ids: np.ndarray,
                                                 node_interact_times: np.ndarray):
        """
        compute source and destination node temporal embeddings.
        :param src_node_ids: ndarray, shape (batch_size, )
        :param dst_node_ids:: ndarray, shape (batch_size, )
        :param node_interact_times: ndarray, shape (batch_size, )
        :return:
        """
        if self.not_embedding:
            node_embeddings = torch.zeros((len(src_node_ids)+len(dst_node_ids),self.output_dim),device=self.device)
        else:
            node_embeddings = self.embedding_module.compute_node_temporal_embeddings(
                node_ids=np.concatenate([src_node_ids, dst_node_ids]),
                src_node_ids=np.tile(src_node_ids, 2),
                dst_node_ids=np.tile(dst_node_ids, 2),
                node_interact_times=np.tile(node_interact_times, 2))
        src_node_embeddings, dst_node_embeddings = node_embeddings[:len(src_node_ids)], node_embeddings[
            len(src_node_ids):]
        return src_node_embeddings, dst_node_embeddings

    def set_neighbor_sampler(self, neighbor_sampler: NeighborSampler):
        """
        set neighbor sampler to neighbor_sampler and reset the random state (for reproducing the results for uniform and time_interval_aware sampling).
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :return:
        """
        if self.embedding_module is not None:
            self.embedding_module.neighbor_sampler = neighbor_sampler
            if self.embedding_module.neighbor_sampler.sample_neighbor_strategy in ['uniform', 'time_interval_aware']:
                assert self.embedding_module.neighbor_sampler.seed is not None
                self.embedding_module.neighbor_sampler.reset_random_state()


class TPNetEmbedding(nn.Module):
    def __init__(self, node_raw_features: torch.Tensor, edge_raw_features: torch.Tensor,
                 neighbor_sampler: NeighborSampler,
                 time_encoder: nn.Module, node_feat_dim: int, edge_feat_dim: int, output_dim: int,
                 time_feat_dim: int, num_layers: int, num_neighbors: int, dropout: float, random_projections: RandomProjectionModule):
        """
        Embedding module of TPNet, which utilizes a multi-layer MLP-Mixer as its backbone.
        :param node_raw_features: Tensor, shape (num_nodes + 1, node_feat_dim)
        :param edge_raw_features: Tensor, shape (num_edges + 1, edge_feat_dim)
        :param neighbor_sampler: NeighborSampler, neighbor sampler
        :param time_encoder: TimeEncoder
        :param node_feat_dim: int, dimension of node features
        :param edge_feat_dim: int, dimension of edge features
        :param time_feat_dim:  int, dimension of time features (encodings)
        :param num_layers: int, number of MLP-Mixer layers
        :param dropout: float, dropout rate
        """
        super(TPNetEmbedding, self).__init__()

        self.node_raw_features = node_raw_features
        self.edge_raw_features = edge_raw_features
        self.neighbor_sampler = neighbor_sampler
        self.time_encoder = time_encoder
        self.node_feat_dim = node_feat_dim
        self.edge_feat_dim = edge_feat_dim
        self.output_dim = output_dim
        self.time_feat_dim = time_feat_dim
        self.num_layers = num_layers
        self.num_neighbors = num_neighbors
        self.dropout = dropout
        self.random_projections = random_projections
        if self.random_projections is None:
            self.random_feature_dim = 0
        else:
            self.random_feature_dim = self.random_projections.pair_wise_feature_dim * 2
        self.projection_layer = nn.Sequential(
            nn.Linear(node_feat_dim + edge_feat_dim + time_feat_dim +
                      self.random_feature_dim, self.output_dim * 2),
            nn.ReLU(), nn.Linear(self.output_dim * 2, self.output_dim))
        self.mlp_mixers = nn.ModuleList([
            MLPMixer(num_tokens=self.num_neighbors, num_channels=self.output_dim,
                     token_dim_expansion_factor=0.5,
                     channel_dim_expansion_factor=4.0, dropout=self.dropout)
            for _ in range(self.num_layers)
        ])

    def compute_node_temporal_embeddings(self, node_ids: np.ndarray, src_node_ids: np.ndarray,
                                         dst_node_ids: np.ndarray, node_interact_times: np.ndarray):
        """
        given memory, node ids node_ids, and the corresponding time node_interact_times, return the temporal embeddings.
        :param node_ids: ndarray, shape (batch_size, ), node ids
        :param node_interact_times: ndarray, shape (batch_size, ), node interaction times
        """

        device = self.node_raw_features.device
        # get temporal neighbors, including neighbor ids, edge ids and time information
        # neighbor_node_ids ndarray, shape (batch_size, num_neighbors)
        # neighbor_edge_ids ndarray, shape (batch_size, num_neighbors)
        # neighbor_times ndarray, shape (batch_size, num_neighbors)
        neighbor_node_ids, neighbor_edge_ids, neighbor_times = \
            self.neighbor_sampler.get_historical_neighbors(node_ids=node_ids,
                                                           node_interact_times=node_interact_times,
                                                           num_neighbors=self.num_neighbors)
        # get node features, shape (batch,num_neighbors,node_feat_dim)
        neighbor_node_features = self.node_raw_features[torch.from_numpy(
            neighbor_node_ids)]
        neighbor_delta_times = torch.from_numpy(
            node_interact_times[:, np.newaxis] - neighbor_times).float().to(device)
        # scale the delta times
        neighbor_delta_times = torch.log(neighbor_delta_times + 1.0)
        # get time encoding, shape (batch,num_neighbors, time_feat_dim)
        neighbor_time_features = self.time_encoder(neighbor_delta_times)
        # get edge features, shape (batch,num_neighors,edge_feat_dim)
        neighbor_edge_features = self.edge_raw_features[torch.from_numpy(
            neighbor_edge_ids)]

        # assign relative encodings for neighbor nodes
        # given a source node u, a destination ndoe v, and a target node w (neighbor of u or v)
        # its relative encoding is [r_{w|u},r_{w|v}], where r_{w|u}/r_{w|v} is the pairwise feature
        # given by the calling the get_pair_wise_feature(w,u)/get_pair_wise_feature(w,v) of the RandomProjectionModule
        if self.random_projections is not None:
            # [2*batch*num_neighbors,random_feature_dim]
            concat_neighbor_random_features = self.random_projections.get_pair_wise_feature(
                src_node_ids=np.tile(neighbor_node_ids.reshape(-1), 2),
                dst_node_ids=np.concatenate(
                    [np.repeat(src_node_ids, self.num_neighbors), np.repeat(dst_node_ids, self.num_neighbors)]))
            # [batch,num_neighbors,random_feature_dim*2]
            neighbor_random_features = torch.cat(
                [concat_neighbor_random_features[:len(node_ids) * self.num_neighbors],
                 concat_neighbor_random_features[len(node_ids) * self.num_neighbors:]],
                dim=1).reshape(len(node_ids), self.num_neighbors, -1)
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features,
                    neighbor_edge_features, neighbor_random_features],
                dim=2)
        else:
            neighbor_combine_features = torch.cat(
                [neighbor_node_features, neighbor_time_features, neighbor_edge_features], dim=2)

        # shape (batch, num_neighbors, node_feat_dim)
        embeddings = self.projection_layer(neighbor_combine_features)
        # mask the pad nodes (i.e., id = 0)
        embeddings.masked_fill(torch.from_numpy(
            neighbor_node_ids == 0)[:, :, None].to(device), 0)
        for mlp_mixer in self.mlp_mixers:
            embeddings = mlp_mixer(embeddings)
        # shape (batch, node_feat_dim)
        embeddings = torch.mean(embeddings, dim=1)

        return embeddings


class FeedForwardNet(nn.Module):

    def __init__(self, input_dim: int, dim_expansion_factor: float, dropout: float = 0.0):
        """
        two-layered MLP with GELU activation function.
        :param input_dim: int, dimension of input
        :param dim_expansion_factor: float, dimension expansion factor
        :param dropout: float, dropout rate
        """
        super(FeedForwardNet, self).__init__()

        self.input_dim = input_dim
        self.dim_expansion_factor = dim_expansion_factor
        self.dropout = dropout

        self.ffn = nn.Sequential(nn.Linear(in_features=input_dim, out_features=int(dim_expansion_factor * input_dim)),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(in_features=int(
                                     dim_expansion_factor * input_dim), out_features=input_dim),
                                 nn.Dropout(dropout))

    def forward(self, x: torch.Tensor):
        """
        feed forward net forward process
        :param x: Tensor, shape (*, input_dim)
        :return:
        """
        return self.ffn(x)


class MLPMixer(nn.Module):

    def __init__(self, num_tokens: int, num_channels: int, token_dim_expansion_factor: float = 0.5,
                 channel_dim_expansion_factor: float = 4.0, dropout: float = 0.0):
        """
        MLP Mixer.
        :param num_tokens: int, number of tokens
        :param num_channels: int, number of channels
        :param token_dim_expansion_factor: float, dimension expansion factor for tokens
        :param channel_dim_expansion_factor: float, dimension expansion factor for channels
        :param dropout: float, dropout rate
        """
        super(MLPMixer, self).__init__()

        self.token_norm = nn.LayerNorm(num_tokens)
        self.token_feedforward = FeedForwardNet(input_dim=num_tokens, dim_expansion_factor=token_dim_expansion_factor,
                                                dropout=dropout)

        self.channel_norm = nn.LayerNorm(num_channels)
        self.channel_feedforward = FeedForwardNet(input_dim=num_channels,
                                                  dim_expansion_factor=channel_dim_expansion_factor,
                                                  dropout=dropout)

    def forward(self, input_tensor: torch.Tensor):
        """
        mlp mixer to compute over tokens and channels
        :param input_tensor: Tensor, shape (batch_size, num_tokens, num_channels)
        :return:
        """
        # mix tokens
        # Tensor, shape (batch_size, num_channels, num_tokens)
        hidden_tensor = self.token_norm(input_tensor.permute(0, 2, 1))
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.token_feedforward(hidden_tensor).permute(0, 2, 1)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + input_tensor

        # mix channels
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_norm(output_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels)
        hidden_tensor = self.channel_feedforward(hidden_tensor)
        # Tensor, shape (batch_size, num_tokens, num_channels), residual connection
        output_tensor = hidden_tensor + output_tensor

        return output_tensor
