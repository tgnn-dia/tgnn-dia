# Evaluation protocol

This file documents the protocol that produced the corpus, precisely enough to
extend it with new models. The implementation lives in this directory; the
protocol-critical pieces are called out below.

## Negative sampling (the alignment mechanism)

`utils/DataLoader.py`, class `ClassicNegativeEdgeSampler`. Every positive test
edge is ranked against 99 negative destinations drawn from the pool of
destinations observed during training (`historical` strategy). The draw is
seeded per evaluation batch from the data itself:

```
batch_seed = (first_source_id * 104729 + first_timestamp) % 2**32
```

Because the seed depends only on the dataset and the batch boundaries, every
model and every random seed scores exactly the same candidate slate for every
edge. This is what makes the corpus rows comparable across architectures.

**Consequence: the evaluation batch size is part of the protocol.** All corpus
runs used batch size 200. A different batch size shifts batch boundaries,
changes every slate, and produces ranks that are not comparable to the corpus.
Extend the corpus only with `--batch_size 200`.

## Rank definition

`utils/evaluate_models_utils.py`. The corpus stores, per edge and model,

```
rank = 1 + #(negatives scored strictly above the positive)
```

Ties do not count against the positive. The MRR metric printed during training
uses the opposite convention (`>=`, mirroring TGB); for models with continuous
scores the two never differ. EdgeBank produces binary scores where ties are
massive, so its corpus column uses expected mid-rank tie-breaking
(`run_edgebank.py`), giving fractional ranks.

## Evaluation order

Within each batch, negatives are scored before positives, so memory-based
models (TGN) cannot absorb a positive edge before scoring that edge's own
candidate slate.

## Datasets and configurations

`utils/DataLoader.py`, `get_link_prediction_data_classic`: chronological
70/15/15 split, discrete-time datasets converted to seconds. Per-dataset model
hyperparameters follow the published configurations in
`utils/load_configs.py`.

## Producing a new corpus column

```
python3 train_link_prediction.py --prefix v2 --dataset_name <ds> \
  --model_name <Model> --load_best_configs --num_runs 3 \
  --eval_neg_strategy historical --save_test_predictions
```

writes `saved_results/v2_link_<ds>_<Model>_seed<k>_test_predictions.csv`
(edge_index, rank), which `../build/build_corpus.py` merges into the corpus
tables.
