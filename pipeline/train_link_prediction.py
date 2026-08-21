# python3 train_link_prediction.py --prefix std --dataset_name tgbl-wiki --model_name TPNet --gpu 0 --load_best_configs --patience 5 --num_epochs 30 --use_random_projection



import logging
import time
import sys
import os
from pathlib import Path

os.environ.setdefault("WANDB_MODE", 'online') # enable / disable wandb logging (env var wins)
os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_INIT_TIMEOUT"] = "120"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":16:8"
project_path = Path(__file__).parent.resolve()
os.environ['WANDB_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CACHE_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_CONFIG_DIR'] = f"{project_path}/wandb"
os.environ['WANDB_DATA_DIR'] = f'{project_path}/wandb'
from tqdm import tqdm
import numpy as np
import warnings
import shutil
import json
import torch
import torch.nn as nn
from models.TGAT import TGAT
from models.MemoryModel import MemoryModel, compute_src_dst_node_time_shifts
from models.CAWN import CAWN
from models.TCL import TCL
from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.TPNet import TPNet, RandomProjectionModule
from models.fusion import Fusion
from models.NAT import NAT
from models.history_state import HistoryState
from models.residual_scorer import ResidualScorer
from models.modules import LinkPredictor_v1, LinkPredictor_v2
from utils.utils import set_thread, set_random_seed, convert_to_gpu, get_parameter_sizes, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from utils.evaluate_models_utils import evaluate_model_link_prediction
from utils.metrics import  WandbLinkLogger
from utils.DataLoader import (get_idx_data_loader, get_link_prediction_data,
                              get_link_prediction_data_classic, ClassicEvaluator, CLASSIC_DATASETS)
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args
from utils.metrics import LossFunction
from tgb.linkproppred.evaluate import Evaluator
import pickle as pk
import wandb

if __name__ == "__main__":
    warnings.filterwarnings('ignore')

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)
    # Fusion with the memory block needs the same memory-bank lifecycle as TGN
    args_my_memory = (args.model_name == 'Fusion' and
                      'memory' in [b.strip() for b in args.my_blocks.split(',')])

    # set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # create file handler that logs debug and higher level messages
    fh = logging.FileHandler(f"./logs/{args.prefix}_link_{args.dataset_name}_{args.model_name}.log", mode="w")
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)

    # get data for training, validation and testing
    if args.dataset_name in CLASSIC_DATASETS:
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name = \
            get_link_prediction_data_classic(dataset_name=args.dataset_name,
                                             eval_neg_strategy=args.eval_neg_strategy,
                                             logger=logger)
    else:
        node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data, eval_neg_edge_sampler, eval_metric_name = \
            get_link_prediction_data(dataset_name=args.dataset_name, logger=logger)

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(data=train_data,
                                                  sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                  time_scaling_factor=args.time_scaling_factor, seed=0)

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(data=full_data, sample_neighbor_strategy=args.sample_neighbor_strategy,
                                                 time_scaling_factor=args.time_scaling_factor, seed=1)

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(src_node_ids=train_data.src_node_ids,
                                                 dst_node_ids=train_data.dst_node_ids,
                                                 interact_times=train_data.node_interact_times,
                                                 last_observed_time=train_data.node_interact_times[0],
                                                 negative_sample_strategy=args.train_negative_sample_strategy,
                                                 seed=None if args.train_negative_sample_strategy == 'random' or
                                                              args.train_negative_sample_strategy == 'new_random'
                                                 else 0)

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(train_data.src_node_ids))),
                                                batch_size=args.batch_size, shuffle=False)
    # reduce the inference batch size of tgbl-wiki and tgbl-review to avoide OOM error
    if args.dataset_name == "tgbl-wiki" or args.dataset_name == 'tgbl-review':
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))), batch_size=20,
                                                  shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))), batch_size=20,
                                                   shuffle=False)
    else:
        val_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(val_data.src_node_ids))),
                                                  batch_size=args.batch_size, shuffle=False)
        test_idx_data_loader = get_idx_data_loader(indices_list=list(range(len(test_data.src_node_ids))),
                                                   batch_size=args.batch_size, shuffle=False)

    evaluator = ClassicEvaluator() if args.dataset_name in CLASSIC_DATASETS else Evaluator(name=args.dataset_name)

    val_metric_all_runs, test_metric_all_runs = [], []

    for run in range(args.num_runs):
        set_random_seed(seed=run, deterministic_alg=args.use_random_projection or args.model_name == 'NAT')
        set_thread(3)

        args.seed = run

        # resume support: a seed whose prediction CSV + metrics JSON already exist
        # on disk completed fully in a previous launch -- skip it so an interrupted
        # multi-seed job only redoes its in-flight seed (opt-in via flag).
        if getattr(args, 'skip_completed_runs', False):
            _done_csv = f"./saved_results/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{run}_test_predictions.csv"
            _done_json = f"./saved_results/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{run}.json"
            if os.path.exists(_done_csv) and os.path.exists(_done_json):
                print(f"[skip_completed_runs] seed {run} already complete, skipping.", flush=True)
                continue

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f'configuration is {args}')

        # create model
        random_projections = None
        if args.use_random_projection:
            # create the model to maintain the temporal walk matrices
            random_projections = RandomProjectionModule(node_num=node_raw_features.shape[0],
                                                        edge_num=edge_raw_features.shape[0],
                                                        dim_factor=args.rp_dim_factor,
                                                        num_layer=args.rp_num_layer,
                                                        device=args.device, use_matrix=args.rp_use_matrix,
                                                        beginning_time=train_data.node_interact_times[0],
                                                        not_scale=args.rp_not_scale,
                                                        enforce_dim=args.enforce_dim,
                                                        adaptive_lambda=args.adaptive_lambda,
                                                        time_decay_weight=args.rp_time_decay_weight,
                                                        lambda_context=args.lambda_context,
                                                        target_bound_mode=args.target_bound_mode,
                                                        target_bound_fixed=args.target_bound_fixed,
                                                        norm_mode=args.norm_mode,
                                                        lambda_alpha=args.lambda_alpha,
                                                        no_global_decay=args.no_global_decay,
                                                        control_mode=args.control_mode)
        # create model
        if args.model_name == 'TGAT':
            dynamic_backbone = TGAT(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                    output_dim=args.output_dim, num_layers=args.num_layers,
                                    num_heads=args.num_heads, dropout=args.dropout, device=args.device)
        elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
            # four floats that represent the mean and standard deviation of source and destination node time shifts in the training data, which is used for JODIE
            src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                                 train_data.node_interact_times)
            dynamic_backbone = MemoryModel(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                           neighbor_sampler=train_neighbor_sampler, output_dim=args.output_dim,
                                           time_feat_dim=args.time_feat_dim, model_name=args.model_name,
                                           num_layers=args.num_layers, num_heads=args.num_heads,
                                           dropout=args.dropout, src_node_mean_time_shift=src_node_mean_time_shift,
                                           src_node_std_time_shift=src_node_std_time_shift,
                                           dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst,
                                           dst_node_std_time_shift=dst_node_std_time_shift, device=args.device,
                                           beta=args.pint_beta, num_hop=args.pint_hop)
        elif args.model_name == 'TPNet':
            dynamic_backbone = TPNet(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                     neighbor_sampler=train_neighbor_sampler,
                                     time_feat_dim=args.time_feat_dim, output_dim=args.output_dim,
                                     random_projections=None if args.encode_not_rp else random_projections,
                                     num_neighbors=args.num_neighbors, num_layers=args.num_layers, dropout=args.dropout,
                                     device=args.device, not_embedding=args.not_embedding,
                                     time_encoder_type=args.time_encoder)
        elif args.model_name == 'CAWN':
            dynamic_backbone = CAWN(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                    neighbor_sampler=train_neighbor_sampler,
                                    time_feat_dim=args.time_feat_dim, output_dim=args.output_dim,
                                    position_feat_dim=args.position_feat_dim,
                                    walk_length=args.walk_length,
                                    num_walk_heads=args.num_walk_heads, dropout=args.dropout, device=args.device)
        elif args.model_name == 'TCL':
            dynamic_backbone = TCL(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                   neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                   output_dim=args.output_dim, num_layers=args.num_layers, num_heads=args.num_heads,
                                   num_depths=args.num_neighbors + 1, dropout=args.dropout, device=args.device)
        elif args.model_name == 'GraphMixer':
            dynamic_backbone = GraphMixer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                          neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                          output_dim=args.output_dim, num_tokens=args.num_neighbors,
                                          num_layers=args.num_layers, dropout=args.dropout, device=args.device,
                                          learn_time_encoder=args.graphmixer_learn_time)
        elif args.model_name == 'DyGFormer':
            dynamic_backbone = DyGFormer(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                         neighbor_sampler=train_neighbor_sampler,
                                         time_feat_dim=args.time_feat_dim, output_dim=args.output_dim,
                                         channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                         num_layers=args.num_layers, num_heads=args.num_heads, dropout=args.dropout,
                                         max_input_sequence_length=args.max_input_sequence_length, device=args.device,
                                         use_cooccurrence=not args.no_cooccurrence)
        elif args.model_name == 'Fusion':
            my_blocks = tuple(b.strip() for b in args.my_blocks.split(',') if b.strip())
            if 'memory' in my_blocks:
                src_node_mean_time_shift, src_node_std_time_shift, dst_node_mean_time_shift_dst, dst_node_std_time_shift = \
                    compute_src_dst_node_time_shifts(train_data.src_node_ids, train_data.dst_node_ids,
                                                     train_data.node_interact_times)
            else:
                src_node_mean_time_shift = src_node_std_time_shift = dst_node_mean_time_shift_dst = dst_node_std_time_shift = 0.0
            dynamic_backbone = Fusion(node_raw_features=node_raw_features, edge_raw_features=edge_raw_features,
                                       neighbor_sampler=train_neighbor_sampler, time_feat_dim=args.time_feat_dim,
                                       output_dim=args.output_dim, dropout=args.dropout, device=args.device,
                                       blocks=my_blocks, random_projections=random_projections,
                                       num_neighbors=args.num_neighbors, num_layers=args.num_layers,
                                       time_encoder_type=args.time_encoder,
                                       channel_embedding_dim=args.channel_embedding_dim, patch_size=args.patch_size,
                                       max_input_sequence_length=args.max_input_sequence_length, num_heads=args.num_heads,
                                       gate_conf=args.my_gate_conf,
                                       src_node_mean_time_shift=src_node_mean_time_shift if 'memory' in my_blocks else 0.0,
                                       src_node_std_time_shift=src_node_std_time_shift if 'memory' in my_blocks else 1.0,
                                       dst_node_mean_time_shift_dst=dst_node_mean_time_shift_dst if 'memory' in my_blocks else 0.0,
                                       dst_node_std_time_shift=dst_node_std_time_shift if 'memory' in my_blocks else 1.0)
            if args.my_warmstart:
                donor = {'walk': 'TPNet', 'cooc': 'DyGFormer', 'memory': 'TGN'}
                ckpt_paths = {b: f"./saved_models/{args.my_warmstart_prefix}_link_{args.dataset_name}_{donor[b]}_seed{args.seed}.pkl"
                              for b in my_blocks if b in donor}
                dynamic_backbone.load_pretrained(ckpt_paths, freeze=args.my_freeze, logger=logger)
        elif args.model_name == 'NAT':
            dynamic_backbone = NAT(n_feat=node_raw_features, e_feat=edge_raw_features, time_dim=args.time_feat_dim,
                                   output_dim=args.output_dim, num_neighbors=[1] + args.nat_num_neighbors,
                                   dropout=args.dropout,
                                   n_hops=args.num_layers,
                                   ngh_dim=args.nat_ngh_dim, device=args.device)
            dynamic_backbone.set_seed(args.seed)
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        if args.model_name == 'NAT':
            link_predictor = LinkPredictor_v2(input_dim=args.output_dim + dynamic_backbone.self_dim * 2,
                                              hidden_dim=args.output_dim + dynamic_backbone.self_dim * 2,
                                              output_dim=1)
        else:
            link_predictor = LinkPredictor_v1(input_dim1=args.output_dim,
                                              input_dim2=args.output_dim,
                                              hidden_dim=args.output_dim, output_dim=1,
                                              random_projections=None if args.decode_not_rp else random_projections,
                                              not_encode=args.not_encode)
        model = nn.Sequential(dynamic_backbone, link_predictor)
        logger.info(f'model -> {model}')
        logger.info(f'model name: {args.model_name}, #parameters: {get_parameter_sizes(model) * 4} B, '
                    f'{get_parameter_sizes(model) * 4 / 1024} KB, {get_parameter_sizes(model) * 4 / 1024 / 1024} MB.')

        # hard residual scorer setup
        history_state = None
        residual_scorer = None
        hard_use_cf = False
        hard_use_feat_sim = False
        hard_cfg = None
        if args.use_hard_residual:
            hard_use_cf = not args.hard_no_cf
            hard_use_feat_sim = args.hard_feat_sim and edge_raw_features.shape[1] > 1
            history_state = HistoryState(
                large_gap_value=args.hard_large_gap,
                max_cf_neighbors=args.hard_max_cf_neighbors,
                max_cf_sources=args.hard_max_cf_sources,
                use_edge_feats=hard_use_feat_sim,
                edge_feat_dim=edge_raw_features.shape[1] if hard_use_feat_sim else 0,
            )
            hard_feat_dim = history_state.feature_dim(use_cf=hard_use_cf, use_feat_sim=hard_use_feat_sim)
            residual_scorer = ResidualScorer(
                feat_dim=hard_feat_dim,
                hidden_dim=args.hard_hidden_dim,
                gate_type=args.hard_gate_type,
                tau=args.hard_tau,
            )

        optimizer = create_optimizer(model=model, optimizer_name=args.optimizer, learning_rate=args.learning_rate,
                                     weight_decay=args.weight_decay)

        model = convert_to_gpu(model, device=args.device)

        if args.use_hard_residual and residual_scorer is not None:
            residual_scorer = convert_to_gpu(residual_scorer, device=args.device)
            optimizer.add_param_group({
                'params': list(residual_scorer.parameters()),
                'lr': args.learning_rate,
                'weight_decay': args.weight_decay,
            })
            hard_cfg = {
                'history_state': history_state,
                'scorer': residual_scorer,
                'use_cf': hard_use_cf,
                'use_feat_sim': hard_use_feat_sim,
                'hard_only': args.hard_only,
                'device': args.device,
                'edge_raw_features': edge_raw_features if hard_use_feat_sim else None,
            }

        save_model_path = f"./saved_models/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}.pkl"
        early_stopping = EarlyStopping(patience=args.patience, save_model_path=save_model_path, logger=logger,
                                       model_name=args.model_name)

        loss_func = nn.BCEWithLogitsLoss()
        train_loss_fn = LossFunction(args.train_loss_type)
        wandb_logger = WandbLinkLogger('run', args)
        wandb_logger.watch(model)
        wandb.define_metric("lambda_epoch*", step_metric="batch_in_epoch")
        wandb.define_metric("lambda_val*", step_metric="batch_in_eval")
        wandb.define_metric("lambda_test*", step_metric="batch_in_eval")
        global_batch = 0

        for epoch in range(args.num_epochs):

            model.train()
            if args.use_hard_residual:
                history_state.reset()
                residual_scorer.train()
            if args.model_name in ['DyRep', 'TGAT', 'TGN', 'TPNet', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'PINT', 'Fusion']:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT'] or args_my_memory:
                # reinitialize memory of memory-based models at the start of each epoch
                model[0].memory_bank.__init_memory_bank__()
            if args.model_name == 'NAT':
                model[0].init_ncache()
            if args.use_random_projection:
                # reinitialize the random projections of temporal walk matrices at the start of each epoch
                random_projections.reset_random_projections()

            # store train losses and metrics
            train_losses, train_metrics = [], []
            train_idx_data_loader_tqdm = tqdm(train_idx_data_loader, ncols=120)
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader_tqdm):
                train_data_indices = train_data_indices.numpy()
                batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                    train_data.src_node_ids[train_data_indices], train_data.dst_node_ids[train_data_indices], \
                        train_data.node_interact_times[train_data_indices], train_data.edge_ids[train_data_indices]
                # --- NEU: Burst / Batch-Density Berechnung ---
                # Zeitdifferenz zwischen der letzten und der ersten Kante im aktuellen Batch
                batch_delta_t = batch_node_interact_times[-1] - batch_node_interact_times[0]
                # Kanten pro Zeiteinheit (1e-7 verhindert Division durch Null, falls alle Kanten exakt denselben Timestamp haben)
                batch_density = len(batch_node_interact_times) / (batch_delta_t + 1e-7)
                # ---------------------------------------------

                batch_neg_src_node_ids, batch_neg_dst_node_ids = train_neg_edge_sampler.sample(
                    size=len(batch_src_node_ids) * args.train_neg_num,
                    batch_src_node_ids=batch_src_node_ids,
                    batch_dst_node_ids=batch_dst_node_ids,
                    current_batch_start_time=batch_node_interact_times[0],
                    current_batch_end_time=batch_node_interact_times[-1])
                batch_neg_node_interact_times = np.repeat(batch_node_interact_times, args.train_neg_num)

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ['TGAT', 'CAWN', 'TCL']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
                    # note that negative nodes do not change the memories while the positive nodes change the memories,
                    # we need to first compute the embeddings of negative nodes for memory-based models
                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times,
                                                                          edge_ids=None,
                                                                          edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)

                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids,
                                                                          edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name in ['GraphMixer']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times,
                                                                          num_neighbors=args.num_neighbors,
                                                                          time_gap=args.time_gap)
                elif args.model_name in ['DyGFormer', 'TPNet']:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times)

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times)
                elif args.model_name == 'Fusion':
                    # negatives first so the memory block (if present) only updates on positives;
                    # the walk/cooc blocks ignore edge_ids/edges_are_positive.
                    batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                          dst_node_ids=batch_neg_dst_node_ids,
                                                                          node_interact_times=batch_neg_node_interact_times,
                                                                          edge_ids=None, edges_are_positive=False,
                                                                          num_neighbors=args.num_neighbors)
                    batch_src_node_embeddings, batch_dst_node_embeddings = \
                        model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                          dst_node_ids=batch_dst_node_ids,
                                                                          node_interact_times=batch_node_interact_times,
                                                                          edge_ids=batch_edge_ids, edges_are_positive=True,
                                                                          num_neighbors=args.num_neighbors)
                elif args.model_name == 'NAT':
                    negative_edge_embeddings = \
                        model[0].compute_edge_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                  dst_node_ids=batch_neg_dst_node_ids,
                                                                  node_interact_times=batch_neg_node_interact_times,
                                                                  edge_ids=None,
                                                                  edges_are_positive=False)

                    positive_edge_embeddings = \
                        model[0].compute_edge_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                  dst_node_ids=batch_dst_node_ids,
                                                                  node_interact_times=batch_node_interact_times,
                                                                  edge_ids=batch_edge_ids,
                                                                  edges_are_positive=True)
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                if args.model_name == 'NAT':
                    positive_probabilities = model[1](edge_embeddings=positive_edge_embeddings).squeeze(dim=-1)
                    negative_probabilities = model[1](edge_embeddings=negative_edge_embeddings).squeeze(dim=-1)
                else:
                    positive_probabilities = model[1](src_node_ids=batch_src_node_ids,
                                                      dst_node_ids=batch_dst_node_ids,
                                                      src_node_embeddings=batch_src_node_embeddings,
                                                      dst_node_embeddings=batch_dst_node_embeddings
                                                      ).squeeze(dim=-1)
                    negative_probabilities = model[1](src_node_ids=batch_neg_src_node_ids,
                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                      src_node_embeddings=batch_neg_src_node_embeddings,
                                                      dst_node_embeddings=batch_neg_dst_node_embeddings
                                                      ).squeeze(dim=-1)

                # ── hard residual scoring (features from pre-update history) ──
                if args.use_hard_residual:
                    _pos_ef = edge_raw_features[batch_edge_ids] if hard_use_feat_sim else None
                    pos_feats_np = history_state.compute_features(
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times,
                        edge_feats=_pos_ef, use_cf=hard_use_cf, use_feat_sim=hard_use_feat_sim)
                    neg_feats_np = history_state.compute_features(
                        batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_neg_node_interact_times,
                        edge_feats=None, use_cf=hard_use_cf, use_feat_sim=hard_use_feat_sim)
                    pos_feats_t = torch.tensor(pos_feats_np, device=args.device)
                    neg_feats_t = torch.tensor(neg_feats_np, device=args.device)
                    pos_alpha, pos_hard = residual_scorer(pos_feats_t)
                    neg_alpha, neg_hard = residual_scorer(neg_feats_t)
                    if args.hard_only:
                        positive_probabilities = pos_hard
                        negative_probabilities = neg_hard
                    else:
                        positive_probabilities = positive_probabilities + pos_alpha * pos_hard
                        negative_probabilities = negative_probabilities + neg_alpha * neg_hard
                # ─────────────────────────────────────────────────────────────

                if args.use_random_projection:
                    # update the random projections of temporal walk matrices after observing positive links
                    random_projections.update(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                                              node_interact_times=batch_node_interact_times)
                    if args.model_name == 'TPNet' and random_projections.norm_history:
                        layer_norms = random_projections.norm_history[-1]
                        log_dict = {
                            **{f'lambda_epoch{epoch}/norm_layer{i+1}': n for i, n in enumerate(layer_norms)},
                            f'lambda_epoch{epoch}/lambda_val':    random_projections.lambda_history[-1],
                            f'lambda_epoch{epoch}/target_bound':  random_projections.target_bound_history[-1],
                            f'lambda_epoch{epoch}/batch_density': batch_density,
                            'batch_in_epoch':                     batch_idx,
                        }
                        if random_projections.rho_history:
                            rhos = random_projections.rho_history[-1]
                            log_dict.update({f'lambda_epoch{epoch}/rho_layer{i+1}': r for i, r in enumerate(rhos)})
                        wandb.log(log_dict, step=global_batch)
                global_batch += 1

                # history update AFTER scoring (no leakage)
                if args.use_hard_residual:
                    history_state.update(
                        batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times,
                        edge_feats=edge_raw_features[batch_edge_ids] if hard_use_feat_sim else None)

                # loss (with optional hard-edge upweighting)
                if args.use_hard_residual and args.hard_gamma > 0:
                    pc_raw = torch.expm1(pos_feats_t[:, 0]).clamp(min=0.0).detach()
                    w = (1 + args.hard_gamma * torch.exp(-pc_raw / args.hard_gamma_tau)).detach()
                    w = w / w.mean()  # normalise so overall scale is preserved
                    pos_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        positive_probabilities, torch.ones_like(positive_probabilities),
                        weight=w, reduction='mean')
                    neg_w = w.unsqueeze(1).expand(-1, args.train_neg_num).reshape(-1)
                    neg_bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        negative_probabilities, torch.zeros_like(negative_probabilities),
                        weight=neg_w, reduction='mean')
                    loss = (pos_bce + neg_bce) / 2
                else:
                    loss = train_loss_fn.forward(positive_logits=positive_probabilities,
                                                 negative_logits=negative_probabilities)
                train_losses.append(loss.item())
                input_dict = {
                    "y_pred_pos": positive_probabilities,
                    "y_pred_neg": negative_probabilities.reshape(-1, args.train_neg_num),
                    "eval_metric": [eval_metric_name],
                }
                train_metrics.append({eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})
                train_idx_data_loader_tqdm.set_description(
                    f'Epoch: {epoch + 1}, train for the {batch_idx + 1}-th batch, train loss: {loss.item()}')

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT'] or args_my_memory:
                    # detach the memories and raw messages of nodes in the memory bank after each batch, so we don't back propagate to the start of time
                    model[0].memory_bank.detach_memory_bank()

            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT'] or args_my_memory:
                train_backup_memory_bank = model[0].memory_bank.backup_memory_bank()
                if args.model_name == 'PINT':
                    train_backup_matrix_memory = model[0].matrix_memory.backup_memory()
            if args.model_name == 'NAT':
                train_backup_ncache = model[0].backup_ncache()
            if args.use_random_projection:
                train_backup_random_projections = random_projections.backup_random_projections()
                if args.model_name == 'TPNet':
                    history_path = f"./saved_results/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}_lambda_history_dyn"
                    random_projections.save_history(history_path)
            if args.use_hard_residual:
                train_backup_history = history_state.backup()

            val_losses, val_metrics = evaluate_model_link_prediction(dataset_name = args.dataset_name,
                                                                    model_name=args.model_name,
                                                                     model=model, dtype='val',
                                                                     eval_metric_name=eval_metric_name,
                                                                     neighbor_sampler=full_neighbor_sampler,
                                                                     evaluate_idx_data_loader=val_idx_data_loader,
                                                                     evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                                     evaluator=evaluator,
                                                                     evaluate_data=val_data,
                                                                     loss_func=loss_func,
                                                                     num_neighbors=args.num_neighbors,
                                                                     time_gap=args.time_gap, logger=logger,
                                                                     wandb_prefix=f'lambda_val_epoch{epoch}' if args.model_name == 'TPNet' else None,
                                                                     wandb_step_start=global_batch,
                                                                     hard_residual_config=hard_cfg)
            global_batch += len(val_losses)

            # reload the train memory to ensure the saved model can be reload for validation and test
            if args.model_name in ['JODIE', 'DyRep', 'TGN', 'PINT'] or args_my_memory:
                model[0].memory_bank.reload_memory_bank(train_backup_memory_bank)
                del train_backup_memory_bank
                if args.model_name == 'PINT':
                    model[0].matrix_memory.reload_memory(train_backup_matrix_memory)
                    del train_backup_matrix_memory
            if args.model_name == 'NAT':
                model[0].reload_ncache(train_backup_ncache)
                del train_backup_ncache
            if args.use_random_projection:
                random_projections.reload_random_projections(train_backup_random_projections)
                del train_backup_random_projections
            if args.use_hard_residual:
                history_state.restore(train_backup_history)
                del train_backup_history

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}')
            for metric_name in train_metrics[0].keys():
                logger.info(
                    f'train {metric_name}, {np.mean([train_metric[metric_name] for train_metric in train_metrics]):.4f}')
            logger.info(f'validate loss: {np.mean(val_losses):.4f}')
            for metric_name in val_metrics[0].keys():
                logger.info(
                    f'validate {metric_name}, {np.mean([val_metric[metric_name] for val_metric in val_metrics]):.4f}')
            lambda_epoch_stats = {}
            if args.use_random_projection and args.model_name == 'TPNet' and random_projections.norm_history:
                epoch_len = len(train_idx_data_loader)
                norms_array = np.array(random_projections.norm_history[-epoch_len:])  # (steps, num_layers)
                lambda_epoch_stats = {
                    **{f'lambda/epoch_norm_mean_layer{i+1}': norms_array[:, i].mean() for i in range(norms_array.shape[1])},
                    'lambda/epoch_lambda_mean':       np.mean(random_projections.lambda_history[-epoch_len:]),
                    'lambda/epoch_target_bound_mean': np.mean(random_projections.target_bound_history[-epoch_len:]),
                }
            wandb_logger.log_epoch(train_losses=train_losses, train_metrics=train_metrics, val_losses=val_losses,
                                   val_metrics=val_metrics, epoch=global_batch, extra=lambda_epoch_stats)

            # select the best model based on all the validate metrics
            val_metric_indicator = []
            for metric_name in val_metrics[0].keys():
                val_metric_indicator.append(
                    (metric_name, np.mean([val_metric[metric_name] for val_metric in val_metrics]), True))
            early_stop = early_stopping.step(
                val_metric_indicator, model, args, epoch + 1,
                aux_modules={'residual_scorer': residual_scorer} if args.use_hard_residual else None)

            if early_stop:
                break

        # load the best model
        logger.info(f'---------Load the best parameters at epoch {early_stopping.best_epoch}-------')
        early_stopping.load_checkpoint(
            model,
            map_location=args.device,
            aux_modules={'residual_scorer': residual_scorer} if args.use_hard_residual else None)

        # evaluate the best model
        logger.info(f'---------get final performance on dataset {args.dataset_name}-------')

        # rebuild history from training data for clean final evaluation
        if args.use_hard_residual:
            history_state.reset()
            history_state.update(
                train_data.src_node_ids, train_data.dst_node_ids, train_data.node_interact_times,
                edge_feats=edge_raw_features[train_data.edge_ids] if hard_use_feat_sim else None)
            residual_scorer.eval()

        val_losses, val_metrics = evaluate_model_link_prediction(dataset_name=args.dataset_name,
                                                                model_name=args.model_name,
                                                                 model=model, dtype='val',
                                                                 eval_metric_name=eval_metric_name,
                                                                 neighbor_sampler=full_neighbor_sampler,
                                                                 evaluate_idx_data_loader=val_idx_data_loader,
                                                                 evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                                 evaluator=evaluator,
                                                                 evaluate_data=val_data,
                                                                 loss_func=loss_func,
                                                                 num_neighbors=args.num_neighbors,
                                                                 time_gap=args.time_gap, logger=logger,
                                                                 wandb_prefix='lambda_val_final' if args.model_name == 'TPNet' else None,
                                                                 wandb_step_start=global_batch,
                                                                 hard_residual_config=hard_cfg)
        global_batch += len(val_losses)

        # after val eval, history_state contains train+val edges → correct start for test eval
        test_pred_path = (
            f"./saved_results/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}_test_predictions.csv"
            if args.save_test_predictions else None
        )
        test_hard_pred_path = (
            f"./saved_results/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}_test_hard_predictions.csv"
            if args.save_hard_predictions else None
        )
        test_losses, test_metrics = evaluate_model_link_prediction(dataset_name=args.dataset_name,
                                                                   model_name=args.model_name,
                                                                   model=model, dtype='test',
                                                                   eval_metric_name=eval_metric_name,
                                                                   neighbor_sampler=full_neighbor_sampler,
                                                                   evaluate_idx_data_loader=test_idx_data_loader,
                                                                   evaluate_neg_edge_sampler=eval_neg_edge_sampler,
                                                                   evaluator=evaluator,
                                                                   evaluate_data=test_data,
                                                                   loss_func=loss_func,
                                                                   num_neighbors=args.num_neighbors,
                                                                   time_gap=args.time_gap, logger=logger,
                                                                   wandb_prefix='lambda_test' if args.model_name == 'TPNet' else None,
                                                                   wandb_step_start=global_batch,
                                                                   save_predictions_path=test_pred_path,
                                                                   hard_residual_config=hard_cfg,
                                                                   save_hard_predictions_path=test_hard_pred_path)
        global_batch += len(test_losses)

        # store the evaluation metrics at the current run
        val_metric_dict, test_metric_dict = {}, {}

        logger.info(f'validate loss: {np.mean(val_losses):.4f}')
        for metric_name in val_metrics[0].keys():
            average_val_metric = np.mean([val_metric[metric_name] for val_metric in val_metrics])
            logger.info(f'validate {metric_name}, {average_val_metric:.4f}')
            val_metric_dict[metric_name] = average_val_metric

        logger.info(f'test loss: {np.mean(test_losses):.4f}')
        for metric_name in test_metrics[0].keys():
            average_test_metric = np.mean([test_metric[metric_name] for test_metric in test_metrics])
            logger.info(f'test {metric_name}, {average_test_metric:.4f}')
            test_metric_dict[metric_name] = average_test_metric

        single_run_time = time.time() - run_start_time
        logger.info(
            f'Run {run + 1} cost {single_run_time:.2f} seconds. Maximum GPU memory usage is {torch.cuda.max_memory_allocated(device=args.device) / 1024 / 1024:.0f} MB')
        wandb_logger.log_run(val_losses=val_losses, val_metrics=val_metrics, test_losses=test_losses,
                             test_metrics=test_metrics)
        wandb_logger.finish()

        val_metric_all_runs.append(val_metric_dict)
        test_metric_all_runs.append(test_metric_dict)

        # save model result
        result_json = {
            "validate metrics": {metric_name: f'{val_metric_dict[metric_name]}' for metric_name in
                                 val_metric_dict},

            "test metrics": {metric_name: f'{test_metric_dict[metric_name]}' for metric_name in
                             test_metric_dict}
        }

        result_json = json.dumps(result_json, indent=4)

        save_result_path = f"./saved_results/{args.prefix}_link_{args.dataset_name}_{args.model_name}_seed{args.seed}.json"

        with open(save_result_path, 'w') as file:
            file.write(result_json)

    # store the average metrics at the log of the last run
    if args.num_runs > 1 and len(val_metric_all_runs) > 0:
        logger.info(f'-----------metrics over {args.num_runs} runs-----------')
        wandb_logger = WandbLinkLogger('summary', args)
        for metric_name in val_metric_all_runs[0].keys():
            logger.info(
                f'average validate {metric_name}, {np.mean([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs]):.4f} '
                f'± {np.std([val_metric_single_run[metric_name] for val_metric_single_run in val_metric_all_runs], ddof=1):.4f}')
            logger.info(
                f'average test {metric_name}, {np.mean([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs]):.4f} '
                f'± {np.std([test_metric_single_run[metric_name] for test_metric_single_run in test_metric_all_runs], ddof=1):.4f}')
        wandb_logger.log_final(val_metrics=val_metric_all_runs, test_metrics=test_metric_all_runs)

    sys.exit()
