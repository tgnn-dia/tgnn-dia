# Training and evaluation pipeline

The code that produced the corpus. It builds on the TGB evaluation variant of
the publicly available TPNet codebase (which in turn builds on DyGLib) and
inherits its one-versus-many ranking protocol from TGB; the additions made for
this work are:

- `utils/DataLoader.py`: the classic (non-TGB) dataset layer
  (`get_link_prediction_data_classic`) and the deterministic
  `ClassicNegativeEdgeSampler` that makes predictions alignable across models
  (see `PROTOCOL.md`)
- `utils/evaluate_models_utils.py`: per-edge rank recording
  (`--save_test_predictions`), evaluated negatives-before-positives
- `models/fusion.py`: the gated fusion over frozen experts (the paper's *Fusion* model, `--model_name Fusion`)
- `run_edgebank.py`: the EdgeBank baseline with expected mid-rank tie-breaking

Running the pipeline requires the graph datasets (standard DyGLib `DG_data/`
layout) and a GPU; none of this is needed to use the corpus or reproduce the
paper's analyses, which run from `../corpus` alone.
