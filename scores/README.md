# Per-candidate scores

One npz file per dataset, expert and seed: `{dataset}_{model}_call{seed}.npz`
for model in TPNet, DyGFormer, TGN and seed in 0, 1, 2. Arrays:

  edge_index  (n,)      test-edge identifiers, matching corpus/{dataset}.csv.gz
  pos         (n,)      raw model score of the true destination
  neg         (n, 99)   raw scores of the 99 negative candidates

The candidate slates are identical across models and seeds, so score-level
fusion across models is exact.

The scores are distributed as release assets, one tar per dataset
(`scores_{dataset}.tar`, 18 MB to 370 MB). Download the ones you need from
<https://github.com/tgnn-dia/tgnn-dia/releases/tag/v1.0> and unpack them into
this directory:

```
for f in scores_*.tar; do tar xf "$f"; done
```

then run `paper/ensembles.py`. Only Table VII needs these files; every
other analysis runs from the corpus alone.
