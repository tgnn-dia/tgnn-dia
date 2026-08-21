"""Per-edge diagnostic framework for CTDG link prediction.

Three small modules:
  corpus      load the aligned prediction tables, novel mask, solved sets
  quantities  set operations over solved sets (composition, oracle, coverage)
  scores      per-candidate score access and score-level fusion primitives

The scripts in ../paper are worked examples: every table and figure of the
paper is a short composition of these functions.
"""
from .corpus import (
    CORPUS, DATASETS, MODELS, SEEDS, K, DESCRIPTORS,
    load, novel, solved,
)
from .quantities import (
    hit_matrix, solved_sets, composition, best_single, oracle,
    drop_weakest_gain, pairwise_coverage, jaccard,
)
from .scores import SCORES, load_scores, sigmoid, hit10, slate_ranks
