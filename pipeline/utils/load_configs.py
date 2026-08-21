import argparse
import sys
import torch
import distutils.util


def get_link_prediction_args(is_evaluation: bool = False):
    """
    get the args for the link prediction task
    :param is_evaluation: boolean, whether in the evaluation process
    :return:
    """
    # arguments
    parser = argparse.ArgumentParser("Interface for the link prediction task")
    parser.add_argument(
        "--prefix", type=str, help="prefix of the experiment", default="test"
    )
    parser.add_argument(
        "--skip_completed_runs", action="store_true", default=False,
        help="skip seeds whose prediction CSV + metrics JSON already exist (resume an interrupted multi-seed job)",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="dataset to be used",
        default="tgbl-wiki",
        choices=[
            # TGB datasets
            "tgbl-wiki", "tgbl-review", "tgbl-coin", "tgbl-comment", "tgbl-flight",
            # classic datasets (loaded from DG_data/)
            "wikipedia", "reddit", "mooc", "lastfm", "enron",
            "socialevo", "uci", "flights", "canparl", "uslegis",
            "untrade", "unvote", "contacts", "lanl", "lanl_full",
        ],
    )
    parser.add_argument("--batch_size", type=int, default=200, help="batch size")
    parser.add_argument(
        "--model_name",
        type=str,
        default="TPNet",
        help="name of the model, note that EdgeBank is only applicable for evaluation",
        choices=[
            "JODIE",
            "DyRep",
            "TGAT",
            "TGN",
            "CAWN",
            "EdgeBank",
            "TCL",
            "GraphMixer",
            "DyGFormer",
            "TimeTop",
            "TPNet",
            "PINT",
            "NAT",
            "Fusion",
        ],
    )
    parser.add_argument("--gpu", type=int, default=0, help="number of gpu to use")
    parser.add_argument(
        "--output_dim", type=int, default=172, help="dimension of the output embedding"
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=20,
        help="number of neighbors to sample for each node",
    )
    parser.add_argument(
        "--sample_neighbor_strategy",
        type=str,
        default="recent",
        choices=["uniform", "recent", "time_interval_aware"],
        help="how to sample historical neighbors",
    )
    parser.add_argument(
        "--time_scaling_factor",
        default=1e-6,
        type=float,
        help="the hyperparameter that controls the sampling preference with time interval, "
        "a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, "
        "it works when sample_neighbor_strategy == time_interval_aware",
    )
    parser.add_argument(
        "--num_walk_heads",
        type=int,
        default=8,
        help="number of heads used for the attention in walk encoder",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=2,
        help="number of heads used in attention layer",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="number of model layers"
    )
    parser.add_argument(
        "--walk_length", type=int, default=1, help="length of each random walk"
    )
    parser.add_argument(
        "--time_gap",
        type=int,
        default=2000,
        help="time gap for neighbors to compute node features",
    )
    parser.add_argument(
        "--time_feat_dim", type=int, default=100, help="dimension of the time embedding"
    )
    parser.add_argument(
        "--position_feat_dim",
        type=int,
        default=172,
        help="dimension of the position embedding",
    )
    parser.add_argument(
        "--edge_bank_memory_mode",
        type=str,
        default="unlimited_memory",
        help="how memory of EdgeBank works",
        choices=["unlimited_memory", "time_window_memory", "repeat_threshold_memory"],
    )
    parser.add_argument(
        "--time_window_mode",
        type=str,
        default="fixed_proportion",
        help="how to select the time window size for time window memory",
        choices=["fixed_proportion", "repeat_interval"],
    )
    parser.add_argument("--patch_size", type=int, default=1, help="patch size")
    parser.add_argument(
        "--channel_embedding_dim",
        type=int,
        default=50,
        help="dimension of each channel embedding",
    )
    parser.add_argument(
        "--max_input_sequence_length",
        type=int,
        default=32,
        help="maximal length of the input sequence of each node",
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--num_epochs", type=int, default=100, help="number of epochs")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["SGD", "Adam", "RMSprop"],
        help="name of optimizer",
    )
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")
    parser.add_argument(
        "--patience", type=int, default=20, help="patience for early stopping"
    )
    parser.add_argument("--num_runs", type=int, default=3, help="number of runs")
    parser.add_argument(
        "--load_best_configs",
        action="store_true",
        default=False,
        help="whether to load the best configurations",
    )
    parser.add_argument("--pint_beta", type=float, default=0.1, help="the beat of PINT")
    parser.add_argument(
        "--pint_hop", type=int, default=3, help="the hop of the PINT walk matrix"
    )
    parser.add_argument(
        "--nat_ngh_dim", type=int, default=4, help="the dimension of NAT ncahche"
    )
    parser.add_argument(
        "--nat_num_neighbors",
        type=int,
        nargs="*",
        default=[32, 16],
        help="a list of neighbor sampling numbers for different hops of NAT",
    )
    parser.add_argument(
        "--top_encode",
        type=str,
        default="continuous",
        choices=["discrete", "continuous"],
        help="the encoder type of timetop",
    )
    parser.add_argument(
        "--top_lam", type=float, default=0.0000001, help="the decay weight of timetop"
    )
    parser.add_argument(
        "--top_hop", type=int, default=1, help="the hop of timetop decoder"
    )
    parser.add_argument(
        "--top_beta", type=float, default=0.01, help="the weight of timetop decoder"
    )
    parser.add_argument(
        "--train_neg_num",
        type=int,
        default=1,
        help="the number of negative edge per postive edge in training",
    )
    parser.add_argument(
        "--train_negative_sample_strategy",
        type=str,
        default="random",
        choices=["random", "historical", "inductive", "new_random"],
        help="strategy for the negative edge sampling",
    )
    parser.add_argument(
        "--train_loss_type",
        type=str,
        default="pointwise",
        choices=["pointwise", "listwise"],
        help="the loss function type of training",
    )
    parser.add_argument(
        "--use_random_projection",
        action="store_true",
        help="whether use the random projection",
    )
    parser.add_argument(
        "--rp_num_layer", type=int, default=2, help="the layer of random projection"
    )
    parser.add_argument(
        "--rp_time_decay_weight",
        type=float,
        default=None,
        help="time decay weight (default: 1e-6, or 1e-7 for tgbl-review when using --load_best_configs)",
    )
    parser.add_argument(
        "--rp_dim_factor",
        type=int,
        default=10,
        help="the dim factor of random feature w.r.t. the node num",
    )
    parser.add_argument(
        "--encode_not_rp", action="store_true", help="whether to user rpnet in encoder"
    )
    parser.add_argument(
        "--no_cooccurrence", action="store_true",
        help="DyGFormer ablation: zero the neighbour co-occurrence channel (remove pairwise-structural signal)",
    )
    parser.add_argument(
        "--graphmixer_learn_time", action="store_true",
        help="GraphMixer ablation: make the (normally fixed) time encoder trainable",
    )
    parser.add_argument(
        "--my_blocks", type=str, default="walk,cooc",
        help="Fusion: comma-separated building blocks to fuse (subset of walk,cooc,memory)",
    )
    parser.add_argument(
        "--my_warmstart", action="store_true",
        help="Fusion: warm-start each block from its pretrained single-model checkpoint",
    )
    parser.add_argument(
        "--my_warmstart_prefix", type=str, default="suite",
        help="Fusion: prefix of the donor checkpoints to warm-start from",
    )
    parser.add_argument(
        "--my_freeze", action="store_true",
        help="Fusion: freeze the warm-started blocks (train only the adapters + gate + decoder)",
    )
    parser.add_argument(
        "--my_gate_conf", action="store_true",
        help="Fusion: feed each block's per-edge compatibility (dot+cosine of src,dst) to the gate",
    )
    parser.add_argument(
        "--decode_not_rp",
        action="store_true",
        help="whether to user rpnet in link decoder",
    )
    parser.add_argument(
        "--rp_not_scale",
        action="store_true",
        help="whether to scale and relu for inner product of random projections",
    )
    parser.add_argument(
        "--not_encode",
        action="store_true",
        help="whether to user node embeddings in link predictor",
    )
    parser.add_argument(
        "--enforce_dim",
        type=int,
        default=-1,
        help="whether specific the dimension of random prjections",
    )
    parser.add_argument(
        "--rp_use_matrix",
        action="store_true",
        help="whether replace the random projection with temporal walk matrices",
    )
    parser.add_argument(
        "--not_embedding",
        action="store_true",
        help="whether to use the embedding model in TPNet",
    )
    parser.add_argument(
        "--eval_neg_strategy",
        type=str,
        default="historical",
        choices=["random", "historical", "inductive"],
        help="negative sampling strategy for evaluation of classic datasets: "
             "'historical' samples from destination nodes seen in training (hard); "
             "'inductive' samples from nodes not seen in training (tests new nodes); "
             "'random' samples uniformly (easy baseline)",
    )
    parser.add_argument(
        "--adaptive_lambda",
        action="store_true",
        default=False,
        help="whether to use the adaptive lambda in TPNet (if False, uses fixed rp_time_decay_weight)",
    )
    parser.add_argument(
        "--lambda_context",
        action="store_true",
        default=False,
        help="whether to append lambda_val as extra input to the pairwise MLP (only meaningful with --adaptive_lambda)",
    )
    parser.add_argument(
        "--target_bound_mode",
        type=str,
        default="sqrt_Nd",
        choices=["fixed", "sqrt_Nd", "sqrt_N_t_d"],
        help="how to compute the target bound for adaptive lambda: "
             "'fixed' uses --target_bound_fixed; "
             "'sqrt_Nd' uses sqrt(N*d) with total node count N (static); "
             "'sqrt_N_t_d' uses sqrt(N(t)*d) where N(t) grows with unique seen nodes",
    )
    parser.add_argument(
        "--target_bound_fixed",
        type=float,
        default=1000.0,
        help="constant target bound value, used only when --target_bound_mode fixed",
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default="frobenius",
        choices=["frobenius", "avg_row", "max_row"],
        help="which norm to use to measure matrix size for adaptive lambda: "
             "'frobenius' = sqrt(Σ a_ij²); "
             "'avg_row' = mean per-node row norm; "
             "'max_row' = max per-node row norm",
    )
    parser.add_argument(
        "--lambda_alpha",
        type=float,
        default=0.01,
        help="feedback strength for lambda adaptation: lambda *= (error_ratio ** alpha). "
             "Smaller values mean slower, smoother adaptation.",
    )
    parser.add_argument(
        "--no_global_decay",
        action="store_true",
        default=False,
        help="replace time-scaled global decay exp(-lambda*delta_t*i) with per-batch decay exp(-lambda*i), "
             "removing sensitivity to irregular inter-event time gaps (useful for tgbl-review)",
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        default="controller",
        choices=["controller", "projection"],
        help="norm-control strategy used when --adaptive_lambda is set: "
             "'controller' adjusts lambda to drive the norm toward the bound (current default); "
             "'projection' directly rescales the matrices onto the norm ball when the norm exceeds the bound",
    )
    parser.add_argument(
        "--time_encoder",
        type=str,
        default="sinusoidal",
        choices=["sinusoidal", "linear"],
        help="time encoder type for TPNet: "
             "'sinusoidal' uses cos(W·t + b) with frequency initialisation (original TPNet); "
             "'linear' uses W·t + b without cosine wrapping (Chung et al., 2025), "
             "avoids information loss from many-to-one sinusoidal mapping",
    )
    parser.add_argument(
        "--save_test_predictions",
        action="store_true",
        default=False,
        help="save test edge index and rank to saved_results/<prefix>_..._test_predictions.csv",
    )

    # ── Hard-residual scorer ────────────────────────────────────────────────────
    parser.add_argument(
        "--use_hard_residual",
        action="store_true",
        default=False,
        help="add a failure-guided low-history residual scorer on top of the main model",
    )
    parser.add_argument(
        "--hard_gate_type",
        type=str,
        default="learnable",
        choices=["learnable", "exp", "mlp", "none"],
        help="gate type for the residual: "
             "'learnable' sigmoid(a - b*log1p(pair_count)); "
             "'exp' exp(-pair_count/tau); "
             "'mlp' sigmoid(MLP(features)); "
             "'none' alpha=1 (no gate, ablation)",
    )
    parser.add_argument(
        "--hard_tau",
        type=float,
        default=5.0,
        help="tau for the exponential gate: alpha = exp(-pair_count / tau)",
    )
    parser.add_argument(
        "--hard_hidden_dim",
        type=int,
        default=64,
        help="hidden dimension of the residual MLP scorer",
    )
    parser.add_argument(
        "--hard_no_cf",
        action="store_true",
        default=False,
        help="disable bipartite CF-support feature in the residual scorer (faster)",
    )
    parser.add_argument(
        "--hard_feat_sim",
        action="store_true",
        default=False,
        help="enable edge-feature cosine-similarity features in the residual scorer "
             "(requires non-trivial edge features; off by default)",
    )
    parser.add_argument(
        "--hard_only",
        action="store_true",
        default=False,
        help="ablation: use only the residual scorer logit, ignore TPNet output",
    )
    parser.add_argument(
        "--hard_gamma",
        type=float,
        default=0.0,
        help="hard-edge loss upweighting strength: weight = 1 + gamma*exp(-pair_count/hard_gamma_tau). "
             "0.0 disables weighting (default).",
    )
    parser.add_argument(
        "--hard_gamma_tau",
        type=float,
        default=5.0,
        help="tau for hard-edge loss upweighting (see --hard_gamma)",
    )
    parser.add_argument(
        "--hard_max_cf_neighbors",
        type=int,
        default=30,
        help="max destinations sampled from out_neighbors[u] for CF support (trade-off: quality vs speed)",
    )
    parser.add_argument(
        "--hard_max_cf_sources",
        type=int,
        default=30,
        help="max sources sampled from in_neighbors[x] for CF support (trade-off: quality vs speed)",
    )
    parser.add_argument(
        "--hard_large_gap",
        type=float,
        default=1e9,
        help="fill value for time-gap features when a pair/node has not been seen before",
    )
    parser.add_argument(
        "--save_hard_predictions",
        action="store_true",
        default=False,
        help="save extended per-edge CSV (features, alpha, logits) alongside --save_test_predictions",
    )

    try:
        args = parser.parse_args()
        args.device = (
            f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu >= 0 else "cpu"
        )
    except:
        parser.print_help()
        sys.exit()

    if args.model_name == "EdgeBank":
        assert is_evaluation, "EdgeBank is only applicable for evaluation!"

    if args.load_best_configs:
        load_link_prediction_best_configs(args=args)
    if args.use_random_projection and args.rp_time_decay_weight is None:                                                          
        args.rp_time_decay_weight = 1e-6 

    return args


def load_link_prediction_best_configs(args: argparse.Namespace):
    """
    load the best configurations for the link prediction task
    :param args: argparse.Namespace
    :return:
    """
    # model specific settings
    if args.model_name in ["TGAT"]:
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.2 if args.dataset_name in ["enron", "canparl", "unvote"] else 0.1
        args.sample_neighbor_strategy = "uniform" if args.dataset_name in ["reddit", "canparl", "untrade"] else "recent"
    elif args.model_name in ["JODIE", "DyRep", "TGN"]:
        args.num_neighbors = 10
        args.num_layers = 1
        if args.model_name == "JODIE":
            if args.dataset_name in ["mooc", "uslegis"]:
                args.dropout = 0.2
            elif args.dataset_name in ["lastfm"]:
                args.dropout = 0.3
            elif args.dataset_name in ["uci", "untrade"]:
                args.dropout = 0.4
            elif args.dataset_name in ["canparl"]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        elif args.model_name == "DyRep":
            args.dropout = 0.0 if args.dataset_name in ["mooc", "lastfm", "enron", "uci", "canparl", "uslegis", "contacts"] else 0.1
        else:  # TGN
            if args.dataset_name in ["mooc", "untrade"]:
                args.dropout = 0.2
            elif args.dataset_name in ["lastfm", "canparl"]:
                args.dropout = 0.3
            elif args.dataset_name in ["enron", "socialevo"]:
                args.dropout = 0.0
            else:
                args.dropout = 0.1
        if args.model_name in ["TGN", "DyRep"]:
            if args.dataset_name in ["canparl"] or (args.model_name == "TGN" and args.dataset_name == "unvote"):
                args.sample_neighbor_strategy = "uniform"
            else:
                args.sample_neighbor_strategy = "recent"
    elif args.model_name == "CAWN":
        args.time_scaling_factor = 1e-6
        args.num_neighbors = 32
        args.dropout = 0.1
        args.sample_neighbor_strategy = "time_interval_aware"
    elif args.model_name == "EdgeBank":
        args.edge_bank_memory_mode = "time_window_memory"
        args.time_window_mode = "fixed_proportion"
    elif args.model_name == "TCL":
        args.num_neighbors = 20
        args.num_layers = 2
        args.dropout = 0.1
        args.sample_neighbor_strategy = "recent"
    elif args.model_name in ["GraphMixer"]:
        args.num_layers = 2
        if args.dataset_name in ["wikipedia"]:
            args.num_neighbors = 30
        elif args.dataset_name in ["reddit", "lastfm"]:
            args.num_neighbors = 10
        else:
            args.num_neighbors = 20
        if args.dataset_name in ["wikipedia", "reddit", "enron"]:
            args.dropout = 0.5
        elif args.dataset_name in ["mooc", "uci", "uslegis"]:
            args.dropout = 0.4
        elif args.dataset_name in ["lastfm", "unvote"]:
            args.dropout = 0.0
        elif args.dataset_name in ["socialevo"]:
            args.dropout = 0.3
        elif args.dataset_name in ["flights", "canparl"]:
            args.dropout = 0.2
        else:
            args.dropout = 0.1
        args.sample_neighbor_strategy = "uniform" if args.dataset_name in ["canparl", "untrade", "unvote"] else "recent"
    elif args.model_name in ["DyGFormer"]:
        args.num_layers = 2
        if args.dataset_name in ["reddit"]:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ["mooc", "enron", "flights", "uslegis", "untrade"]:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ["lastfm"]:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ["canparl"]:
            # DyGLib's tuned value is 2048 (for AP/AUC eval with 1 negative); that is
            # infeasible under our hard-negative MRR eval (~99x the eval-time gather),
            # so we cap at 512 -- the longest value that fits, matching LastFM.
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ["unvote"]:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        if args.dataset_name in ["reddit", "unvote"]:
            args.dropout = 0.2
        elif args.dataset_name in ["enron", "uslegis", "untrade", "contacts"]:
            args.dropout = 0.0
        else:
            args.dropout = 0.1
    elif args.model_name == "TPNet":
        args.rp_num_layer = 2
        if args.dataset_name == "canparl":
            args.sample_neighbor_strategy = "uniform"
    elif args.model_name == "Fusion":
        # fused blocks reuse the donor models' settings; the DyGFormer (cooc) block
        # MUST use the same per-dataset sequence length / patch as its donor so the
        # warm-start matches.
        args.num_layers = 2
        args.num_neighbors = 20
        args.dropout = 0.1
        args.rp_num_layer = 2
        if args.dataset_name in ["reddit"]:
            args.max_input_sequence_length = 64
            args.patch_size = 2
        elif args.dataset_name in ["mooc", "enron", "flights", "uslegis", "untrade"]:
            args.max_input_sequence_length = 256
            args.patch_size = 8
        elif args.dataset_name in ["lastfm"]:
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ["canparl"]:
            # DyGLib's tuned value is 2048 (for AP/AUC eval with 1 negative); that is
            # infeasible under our hard-negative MRR eval (~99x the eval-time gather),
            # so we cap at 512 -- the longest value that fits, matching LastFM.
            args.max_input_sequence_length = 512
            args.patch_size = 16
        elif args.dataset_name in ["unvote"]:
            args.max_input_sequence_length = 128
            args.patch_size = 4
        else:
            args.max_input_sequence_length = 32
            args.patch_size = 1
        assert args.max_input_sequence_length % args.patch_size == 0
        args.sample_neighbor_strategy = "uniform" if args.dataset_name in ["canparl", "untrade", "unvote"] else "recent"
    elif args.model_name == "PINT":
        # number of layers
        args.num_layers = 1
        args.num_neighbors = 20
        # beta
        args.pint_beta = 0.0001
    elif args.model_name == "NAT":
        args.nat_ngh_dim = 4
        args.nat_num_neighbors = [32, 16]
    else:
        raise ValueError(f"Wrong value for model_name {args.model_name}!")

    if args.use_random_projection:
        if args.rp_time_decay_weight is None:
            args.rp_time_decay_weight = 0.0000001 if args.dataset_name == "tgbl-review" else 0.000001
