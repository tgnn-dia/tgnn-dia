# A Per-Edge Diagnostic Framework for CTDG Link Prediction

Companion artifact for *A Per-Edge Diagnostic Framework for Continuous-Time
Graph Learning*. The framework studies architectural behaviour at the level of
individual predictions instead of aggregate metrics. Its instrument is the
**aligned prediction corpus**: for every evaluated model, its prediction on
every test edge, scored against identical negatives, as one row per edge.
Because the rows are aligned across models and seeds, any question about which
edges a model solves, misses, owns, or loses is a set operation over a shared
table. Every table and figure of the paper is such an operation and reproduces
in seconds on a laptop. No model, GPU or graph dataset is needed.

## Repository layout

```
diagnostics/ the framework: corpus access (corpus.py), set-level quantities
             (quantities.py), score-level fusion primitives (scores.py)
corpus/      the instrument: one gzipped CSV per dataset (8 datasets, 9 MB total)
paper/       one script per paper table or figure, each a short composition
             of diagnostics functions, reading only corpus/*.csv.gz
build/       provenance: the script that assembled the corpus from raw
             per-run prediction files
pipeline/    the training and evaluation code that produced the predictions,
             with the exact protocol documented in pipeline/PROTOCOL.md
scores/      per-candidate scores (release asset, ~1 GB; needed only for
             score-level fusion) and their format description
```

## Setup

```
pip install -r requirements.txt        # numpy, pandas, scikit-learn
```

## The corpus

One row per positive test edge, in chronological order. Columns:

| Column | Meaning |
|---|---|
| `edge_index`, `src`, `dst`, `time` | edge identifiers |
| 18 descriptor columns | interpretable edge descriptors computed from the training+validation history only. `recurs_later` is future-derived and used only to label edges, never as a descriptor. `is_repeated == 0` defines the novel-edge subset used throughout the paper. |
| `TPNet_s{0,1,2}`, `DyGFormer_s{0,1,2}`, `TGN_s{0,1,2}`, `GraphMixer_s{0,1,2}` | rank of the true destination among 1 positive + 99 deterministic negative candidates, identical for every model and seed; one column per random seed |
| `Fusion_s{0,1,2}` | the learned gated fusion over frozen experts |
| `EdgeBank` | deterministic memorisation baseline; fractional ranks from expected mid-rank tie-breaking |
| ablation columns | only on the datasets they were run on: `TPNet_nowalk` (enron, canparl), `TPNet_recency`, `TPNet_nowalk_recency` (canparl), `DyGFormer_nocooc`, `GraphMixer_learnedtime`, `TPNet_lineartime` (uci, canparl) |

An edge is *solved* by a model when its rank is at most 10 (hit@10).

## Diagnosing your own model

A new architecture joins the corpus through one evaluation pass and one
appended column, no other model needs to be re-run:

1. Evaluate your model with the code in `pipeline/` under the protocol in
   `pipeline/PROTOCOL.md`. The negative slates are deterministic and seeded per
   evaluation batch, so keeping the evaluation batch size at 200 reproduces the
   exact candidate sets every corpus model was scored against.
2. Append the resulting rank column (`YourModel_s{seed}`) to the dataset's
   corpus table.
3. Add your model's name to `MODELS` in `diagnostics/corpus.py`. Every
   quantity and every script in `paper/` iterates that list, so all
   diagnostics now include your model.

## Writing your own diagnostic

`diagnostics` is the API the paper scripts are built on: `load`, `novel` and
`solved` in `corpus.py`, the set-level quantities of Sec. III-C in
`quantities.py`, and score-fusion primitives in `scores.py`. For example, the
novel edges that only TPNet solves on CanParl:

```python
from diagnostics import load, novel, solved

tab = load("canparl")
others = solved(tab, "DyGFormer", 0) | solved(tab, "TGN", 0) | solved(tab, "GraphMixer", 0)
only = solved(tab, "TPNet", 0) & ~others & novel(tab)
print(only.sum())        # 335 of 5738 novel edges (seed 0)
```

Cross-referencing such sets with the descriptor columns (which edges are they?)
or with another model's solved set (does your change win them or lose them?) is
the intended use. The scripts in `paper/` are worked examples of exactly this.

## Reproducing the paper

Run each script from `paper/`. Spot-check values are given so you can
verify you reproduce the paper exactly.

| Paper | Script | What it computes | Spot check |
|---|---|---|---|
| Table I | – | the descriptor definitions; each descriptor is a corpus column | – |
| Fig. 2A/B | `complementarity.py` | all-correct / contested / none fractions, best single vs best-of-4 oracle, oracle gain, drop-weakest control | CanParl: 79.0% contested, gain +24.9, drop-weakest +24.3 |
| Fig. 2C | `complementarity.py [dataset]` | pairwise coverage matrix (% of row's correct novel edges the column misses) | CanParl row TPNet: 32 / 64 / 27 |
| Table II | `seed_control.py` | disagreement between architectures vs between seeds | CanParl: 44.9% vs 13.3% |
| Table III | `seed_control.py` | 3-seed oracle of the best single architecture vs best three-architecture oracle (the trio is chosen by coverage, not individual accuracy: on UCI, USLegis and Enron it keeps TGN, the individually weakest model there) | CanParl: 0.838 vs 0.957 (+11.9) |
| Table IV | `seed_control.py` | stable architecture-specific edges (solved at all seeds by one model, missed at all seeds by another) | CanParl 28.8%, MOOC 32.2% |
| Fig. 3 | `rankings.py` | novel-edge hit@10 per architecture and dataset, with EdgeBank | UNtrade: TPNet 0.355, DyGFormer 0.059 |
| Table V | – | the architectural block inventory; the ablated blocks correspond to the ablation columns | – |
| Fig. 4A–C | `ablations.py` | within-model ablations: walk projections, co-occurrence, time encoders | Enron TPNet 0.623 vs no-walk 0.398; UCI DyGFormer 0.794 vs no-cooc 0.441 |
| Fig. 4D | `ablations.py` | convergence: change in solved-set overlap (Jaccard) with each competitor | CanParl TPNet-nowalk vs GraphMixer: +0.239 |
| (Sec. V) | `ablations.py` | descriptor-null check: no descriptor separates lost from retained edges consistently | max standardised gap 0.79, sign-flipping across datasets |
| Table VI | `router.py` | descriptor-based router (chronological half split) and pairwise ownership AUC | CanParl +37%, USLegis +54%; mean AUC 0.665, Enron 0.507 |
| Table VII | `ensembles.py [scores_dir]` | mean / max / Borda score fusion vs the learned gate | CanParl: mean 0.820, gate 0.803; USLegis: rank 0.668 |
| (Sec. VI-B) | `seed_ensemble.py [scores_dir]` | seed-ensemble control: fusing three seeds of one architecture, separating variance reduction from complementarity | USLegis: TPNet seed-rank 0.665 vs cross-architecture 0.668 |

`ensembles.py` and `seed_ensemble.py` are the only scripts that need more than
the corpus: download the per-candidate scores from
<https://github.com/tgnn-dia/tgnn-dia/releases/tag/v1.0> and unpack
them into `scores/` (format in `scores/README.md`). The candidate slates are
identical across models, so score-level fusion is exact.

## Provenance

`build/build_corpus.py` assembled the corpus from the raw per-run prediction
files. Models were trained and evaluated with DyGLib-based tooling under the
hard historical-negative protocol, using the published per-dataset
hyperparameter configurations, three random seeds per model.
