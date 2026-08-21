import csv
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import logging
import wandb
from utils.utils import NeighborSampler
from utils.DataLoader import Data
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.linkproppred.evaluate import Evaluator
from typing import Callable


def evaluate_model_link_prediction(dataset_name:str,model_name: str, model: nn.Module, dtype: str, eval_metric_name: str,
                                   neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data,
                                   evaluator: Evaluator, loss_func: nn.Module, num_neighbors: int = 20,
                                   time_gap: int = 2000, logger: logging.Logger = None,
                                   wandb_prefix: str = None, wandb_step_start: int = 0,
                                   save_predictions_path: str = None,
                                   hard_residual_config: dict = None,
                                   save_hard_predictions_path: str = None):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :param logger:
    :return:
    """

    if model_name in ['DyRep', 'TGAT', 'TGN', 'TPNet', 'CAWN', 'TCL', 'GraphMixer', 'DyGFormer', 'PINT', 'Fusion']:
        # evaluation phase use all the graph information
        model[0].set_neighbor_sampler(neighbor_sampler)
    if model_name == 'NAT':
        model[0].reset_random_state()

    model.eval()
    if hard_residual_config is not None:
        hard_residual_config['scorer'].eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        prediction_records = [] if save_predictions_path else None
        hard_records = [] if save_hard_predictions_path else None
        alpha_log_data = [] if (hard_residual_config is not None and logger is not None) else None
        num_eval_batches = len(evaluate_idx_data_loader)
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[
                    evaluate_data_indices]

            batch_delta_t = batch_node_interact_times[-1] - batch_node_interact_times[0]
            batch_density = len(batch_node_interact_times) / (batch_delta_t + 1e-7)

            batch_neg_dst_node_ids = evaluate_neg_edge_sampler.query_batch(pos_src=batch_src_node_ids - 1,
                                                                           pos_dst=batch_dst_node_ids - 1,
                                                                           pos_timestamp=batch_node_interact_times,
                                                                           split_mode=dtype)
            # one edge of tgbl-wiki only has 998 negative samples (others has 999 negative samples), 
            # to avoide mismatched shape when perform batch computation,we pad it with an empty edge
            if dataset_name == 'tgbl-wiki':
                batch_neg_dst_node_ids = [x if len(x)==999 else x+[-1] for x in batch_neg_dst_node_ids]
            
            batch_neg_dst_node_ids = (np.array(
                batch_neg_dst_node_ids, dtype=batch_src_node_ids.dtype) + 1).reshape(-1)
            num_negative_samples_per_node = len(
                batch_neg_dst_node_ids) // len(batch_src_node_ids)
            assert num_negative_samples_per_node * \
                len(batch_src_node_ids) == len(batch_neg_dst_node_ids)

            batch_neg_src_node_ids = np.repeat(
                batch_src_node_ids, repeats=num_negative_samples_per_node)
            batch_neg_node_interact_times = np.repeat(
                batch_node_interact_times, repeats=num_negative_samples_per_node)

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['TGAT', 'CAWN', 'TCL']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_neg_node_interact_times,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['JODIE', 'DyRep', 'TGN', 'PINT']:
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
                                                                      num_neighbors=num_neighbors)

                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids,
                                                                      edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_neg_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
            elif model_name in ['DyGFormer', 'TPNet']:
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
            elif model_name == 'Fusion':
                # negatives first so the memory block (if present) only updates on positives
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_neg_node_interact_times,
                                                                      edge_ids=None, edges_are_positive=False,
                                                                      num_neighbors=num_neighbors)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      edge_ids=batch_edge_ids, edges_are_positive=True,
                                                                      num_neighbors=num_neighbors)
            elif model_name == 'NAT':
                negative_edge_embeddings = \
                    model[0].compute_edge_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                              dst_node_ids=batch_neg_dst_node_ids,
                                                              node_interact_times=batch_neg_node_interact_times,
                                                              edge_ids=None,
                                                              edges_are_positive=False)

                positive_edge_embeddings = \
                    model[0].compute_edge_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                              dst_node_ids=batch_dst_node_ids,
                                                              node_interact_times=batch_neg_node_interact_times,
                                                              edge_ids=batch_edge_ids,
                                                              edges_are_positive=True)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            # get positive and negative probabilities, shape (batch_size, )
            if model_name == 'NAT':
                positive_probabilities = model[1](
                    edge_embeddings=positive_edge_embeddings).squeeze(dim=-1)
                negative_probabilities = model[1](
                    edge_embeddings=negative_edge_embeddings).squeeze(dim=-1)
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

            # ── hard residual scorer ───────────────────────────────────────────
            if hard_residual_config is not None:
                _hs      = hard_residual_config['history_state']
                _scorer  = hard_residual_config['scorer']
                _use_cf  = hard_residual_config['use_cf']
                _use_fs  = hard_residual_config['use_feat_sim']
                _h_only  = hard_residual_config['hard_only']
                _dev     = hard_residual_config['device']
                _efeats  = hard_residual_config.get('edge_raw_features')

                _pos_ef = _efeats[batch_edge_ids] if _use_fs and _efeats is not None else None
                pos_feats_np = _hs.compute_features(
                    batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times,
                    edge_feats=_pos_ef, use_cf=_use_cf, use_feat_sim=_use_fs)
                neg_feats_np = _hs.compute_features(
                    batch_neg_src_node_ids, batch_neg_dst_node_ids, batch_neg_node_interact_times,
                    edge_feats=None, use_cf=_use_cf, use_feat_sim=_use_fs)

                pos_feats_t = torch.tensor(pos_feats_np, device=_dev)
                neg_feats_t = torch.tensor(neg_feats_np, device=_dev)
                pos_alpha, pos_hard = _scorer(pos_feats_t)
                neg_alpha, neg_hard = _scorer(neg_feats_t)

                if alpha_log_data is not None:
                    _pc_for_log = np.expm1(pos_feats_np[:, 0])
                    _al_for_log = pos_alpha.cpu().numpy()
                    alpha_log_data.extend(zip(_pc_for_log.tolist(), _al_for_log.tolist()))

                tpnet_logit_pos = positive_probabilities
                if _h_only:
                    positive_probabilities = pos_hard
                    negative_probabilities = neg_hard
                else:
                    positive_probabilities = positive_probabilities + pos_alpha * pos_hard
                    negative_probabilities = negative_probabilities + neg_alpha * neg_hard
            # ──────────────────────────────────────────────────────────────────

            if prediction_records is not None:
                neg_reshaped = negative_probabilities.reshape(len(evaluate_data_indices), num_negative_samples_per_node)
                ranks = 1 + (neg_reshaped > positive_probabilities.unsqueeze(1)).sum(dim=1).cpu().numpy()
                for idx, rank in zip(evaluate_data_indices, ranks):
                    prediction_records.append((int(idx), int(rank)))

            if model[1].random_projections is not None:
                model[1].random_projections.update(src_node_ids=batch_src_node_ids, dst_node_ids=batch_dst_node_ids,
                                                   node_interact_times=batch_node_interact_times)
                if wandb_prefix is not None and batch_idx < num_eval_batches - 1:
                    rp = model[1].random_projections
                    layer_norms = rp._compute_norm()
                    log_dict = {
                        **{f'{wandb_prefix}/norm_layer{i+1}': n for i, n in enumerate(layer_norms)},
                        f'{wandb_prefix}/lambda_val':    rp.lambda_val.item(),
                        f'{wandb_prefix}/target_bound':  rp.target_bound,
                        f'{wandb_prefix}/batch_density': batch_density,
                        'batch_in_eval':                 batch_idx,
                    }
                    if rp.rho_history:
                        rhos = rp.rho_history[-1]
                        log_dict.update({f'{wandb_prefix}/rho_layer{i+1}': r for i, r in enumerate(rhos)})
                    wandb.log(log_dict, step=wandb_step_start + batch_idx)

            # history update AFTER scoring (no leakage)
            if hard_residual_config is not None:
                _hs     = hard_residual_config['history_state']
                _efeats = hard_residual_config.get('edge_raw_features')
                _use_fs = hard_residual_config['use_feat_sim']
                _hs.update(batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times,
                            edge_feats=_efeats[batch_edge_ids] if _use_fs and _efeats is not None else None)

            # extended hard-predictions record
            if hard_records is not None and hard_residual_config is not None:
                _pc_raw  = np.expm1(pos_feats_np[:, 0])
                _pg_raw  = np.expm1(pos_feats_np[:, 2])
                _sc_raw  = np.expm1(pos_feats_np[:, 3])
                _dc_raw  = np.expm1(pos_feats_np[:, 4])
                _sg_raw  = np.expm1(pos_feats_np[:, 5])
                _dg_raw  = np.expm1(pos_feats_np[:, 6])
                _cf_raw  = np.expm1(pos_feats_np[:, 11]) if hard_residual_config['use_cf'] else np.zeros(len(pos_feats_np))
                _alpha_v = pos_alpha.cpu().numpy()
                _hard_v  = pos_hard.cpu().numpy()
                _tpnet_v = tpnet_logit_pos.cpu().numpy()
                _final_v = positive_probabilities.cpu().numpy()
                neg_rs   = negative_probabilities.reshape(len(evaluate_data_indices), num_negative_samples_per_node)
                _ranks_h = 1 + (neg_rs > positive_probabilities.unsqueeze(1)).sum(dim=1).cpu().numpy()
                for j, idx in enumerate(evaluate_data_indices):
                    hard_records.append((
                        int(idx),
                        int(batch_src_node_ids[j]),
                        int(batch_dst_node_ids[j]),
                        float(batch_node_interact_times[j]),
                        int(_ranks_h[j]),
                        float(_pc_raw[j]),
                        float(pos_feats_np[j, 1]),
                        float(_pg_raw[j]),
                        float(_sc_raw[j]),
                        float(_dc_raw[j]),
                        float(_sg_raw[j]),
                        float(_dg_raw[j]),
                        float(_cf_raw[j]),
                        float(_alpha_v[j]),
                        float(_tpnet_v[j]),
                        float(_hard_v[j]),
                        float(_final_v[j]),
                    ))

            predicts = torch.cat(
                [positive_probabilities, negative_probabilities], dim=0)
            labels = torch.cat([torch.ones_like(positive_probabilities), torch.zeros_like(negative_probabilities)],
                               dim=0)
            loss = loss_func(input=predicts, target=labels)

            evaluate_losses.append(loss.item())

            # evaluate_metric_name = []
            input_dict = {
                "y_pred_pos": positive_probabilities,
                "y_pred_neg": negative_probabilities.reshape(-1, num_negative_samples_per_node),
                "eval_metric": [eval_metric_name],
            }
            evaluate_metrics.append(
                {eval_metric_name: evaluator.eval(input_dict)[eval_metric_name]})

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    if alpha_log_data:
        pcs = np.array([x[0] for x in alpha_log_data])
        als = np.array([x[1] for x in alpha_log_data])
        m0  = pcs == 0
        ml  = (pcs > 0) & (pcs < 6)
        mh  = pcs >= 6
        logger.info(
            f'[hard residual alpha] unseen(pc=0) n={m0.sum()} mean={als[m0].mean():.4f}; '
            f'rare(0<pc<6) n={ml.sum()} mean={als[ml].mean():.4f}; '
            f'common(pc>=6) n={mh.sum()} mean={als[mh].mean():.4f}'
        )

    if prediction_records is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_predictions_path)), exist_ok=True)
        with open(save_predictions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['edge_index', 'rank'])
            writer.writerows(prediction_records)

    if hard_records is not None:
        os.makedirs(os.path.dirname(os.path.abspath(save_hard_predictions_path)), exist_ok=True)
        with open(save_hard_predictions_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'edge_index', 'src', 'dst', 'time', 'rank',
                'pair_count', 'pair_seen', 'pair_gap',
                'src_count', 'dst_count', 'src_gap', 'dst_gap', 'cf_support',
                'alpha', 'tpnet_logit', 'hard_logit', 'final_logit',
            ])
            writer.writerows(hard_records)

    return evaluate_losses, evaluate_metrics


def evaluate_sNet_link_prediction(model: nn.Module, dtype: str, eval_metric_name: str,
                                  neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                  evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data,
                                  evaluator: Evaluator, get_eval_loss: Callable, logger: logging.Logger = None):
    """
    evaluate models on the link prediction task
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_type: str, type of loss function
    :param logger:
    :return:
    """
    model.set_neighbor_sampler(neighbor_sampler)
    model.eval()

    with torch.no_grad():
        # store evaluate losses and metrics
        evaluate_losses, evaluate_metrics = [], []
        evaluate_idx_data_loader_tqdm = tqdm(
            evaluate_idx_data_loader, ncols=120)
        for batch_idx, evaluate_data_indices in enumerate(evaluate_idx_data_loader_tqdm):
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_pos_src_node_ids, batch_pos_dst_node_ids, batch_pos_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices], evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[
                    evaluate_data_indices]

            batch_neg_dst_node_ids = evaluate_neg_edge_sampler.query_batch(pos_src=batch_pos_src_node_ids - 1,
                                                                           pos_dst=batch_pos_dst_node_ids - 1,
                                                                           pos_timestamp=batch_pos_node_interact_times,
                                                                           split_mode=dtype)
            batch_neg_dst_node_ids = (np.array(
                batch_neg_dst_node_ids, dtype=batch_pos_src_node_ids.dtype) + 1).reshape(-1)
            num_negative_samples_per_node = len(
                batch_neg_dst_node_ids) // len(batch_pos_src_node_ids)
            assert num_negative_samples_per_node * \
                len(batch_pos_src_node_ids) == len(batch_neg_dst_node_ids)
            batch_neg_src_node_ids = np.repeat(
                batch_pos_src_node_ids, repeats=num_negative_samples_per_node)
            batch_neg_node_interact_times = np.repeat(
                batch_pos_node_interact_times, repeats=num_negative_samples_per_node)

            batch_src_node_ids = np.concatenate(
                [batch_pos_src_node_ids, batch_neg_src_node_ids])
            batch_dst_node_ids = np.concatenate(
                [batch_pos_dst_node_ids, batch_neg_dst_node_ids])
            batch_node_interact_times = np.concatenate(
                [batch_pos_node_interact_times, batch_neg_node_interact_times])

            selected_models, logits = model.compute_logits(src_node_ids=batch_src_node_ids,
                                                           dst_node_ids=batch_dst_node_ids,
                                                           node_interact_times=batch_node_interact_times)

            if model.memory_model is not None:
                model.memory_model.update(src_node_ids=batch_pos_src_node_ids, dst_node_ids=batch_pos_dst_node_ids,
                                          node_interact_times=batch_pos_node_interact_times)

            labels = torch.cat([torch.ones(len(batch_pos_src_node_ids), device=logits.device),
                                torch.zeros(len(batch_neg_src_node_ids), device=logits.device)], dim=0)
            loss = get_eval_loss(logits=logits, labels=labels)
            evaluate_losses.append(loss.item())

            # evaluate_metric_name = []
            input_dict = {
                "y_pred_pos": logits[:len(batch_pos_src_node_ids)],
                "y_pred_neg": logits[len(batch_pos_src_node_ids):].reshape(-1, num_negative_samples_per_node),
                "eval_metric": [eval_metric_name],
            }
            evaluate_metrics.append(
                {eval_metric_name: evaluator.eval(input_dict)[eval_metric_name],
                 'ratio': 1-np.sum(selected_models)/len(selected_models)})

            evaluate_idx_data_loader_tqdm.set_description(
                f'evaluate for the {batch_idx + 1}-th batch, evaluate loss: {loss.item()}')

    return evaluate_losses, evaluate_metrics
