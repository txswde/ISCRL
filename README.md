# ISCRL

Implementation of **Interpretable Self-supervised Contrastive Reinforcement
Learning for Unsupervised Video Summarization**.

The repository follows the revised manuscript: a SimCLR warm-up learns
invariant frame representations from pre-extracted GoogLeNet pool5 features;
a dual-branch recurrent policy fuses original and invariant feature spaces;
REINFORCE is optimized with the dual-space reward and the Adaptive
Intervention Mechanism (AIM); and policy-conditioned temporal/spatial
explanations are produced with self-attention and smoothed Grad-CAM++.

## What is implemented

- Feature-level SimCLR using Gaussian noise (`sigma=0.03`), feature masking
  (`p=0.20`), and local temporal jitter (`+/-1` sampled frame).
- Projection head `1024-512-128`, temperature `0.5`, and 50 contrastive
  warm-up epochs.
- Original/invariant branches with branch-specific projections,
  bidirectional GRUs, self-attention, residual connections, and learnable
  linear fusion.
- Dual reward
  `R_dual = (1-beta) R_original + beta R_invariant`, where each single-space
  reward is diversity plus representativeness; `beta=0.10` and `epsilon=20`.
- Per-video EMA baselines and AIM with `lambda=0.10`, intervention range
  `[0.10, 0.50]`, increase/decrease factors `1.10/0.90`, and base clipping
  constant `5`.
- A 300-epoch RL phase with learning rate and weight decay of `1e-5`, with the
  learning rate halved every 30 epochs.
- Temporal explanation from the column mean of the fused attention matrix and
  spatial explanation from the frame-selection probability back-propagated to
  the last fixed GoogLeNet convolutional feature maps.

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Data

The code expects the conventional HDF5 files containing 1024-dimensional
GoogLeNet pool5 features sampled at 2 fps. Each video group must include:

```text
features, change_points, n_frames, n_frame_per_seg,
picks, user_summary, gtscore
```

Place files under `datasets/` or pass their paths with repeated `--dataset`
arguments. Dataset files and checkpoints are intentionally excluded from Git.
SumMe is available from the [SumMe project](https://gyglim.github.io/me/vsum/index.html),
and TVSum data and annotations are available from the
[official TVSum repository](https://github.com/yalesong/tvsum). The repository
does not redistribute either dataset.

`--userscore` is optional. When supplied, it must point to an HDF5 file with a
`user_scores` dataset under each video key; Spearman's rho and Kendall's tau
are then reported in addition to F1.

## Training

The manuscript defaults are built into the command-line interface. The
examples below therefore do not need to restate the 50/300 schedule.

Canonical SumMe:

```bash
python main.py \
  --dataset datasets/eccv16_dataset_summe_google_pool5.h5 \
  --split splits/summe_splits.json --split-id 0 --metric summe \
  --save-dir log/summe_canonical_split0
```

Canonical TVSum:

```bash
python main.py \
  --dataset datasets/eccv16_dataset_tvsum_google_pool5.h5 \
  --split splits/tvsum_splits.json --split-id 0 --metric tvsum \
  --save-dir log/tvsum_canonical_split0
```

Augmented SumMe uses target training videos plus OVP and YouTube:

```bash
python main.py \
  --dataset datasets/eccv16_dataset_summe_google_pool5.h5 \
  --dataset datasets/eccv16_dataset_tvsum_google_pool5.h5 \
  --dataset datasets/eccv16_dataset_ovp_google_pool5.h5 \
  --dataset datasets/eccv16_dataset_youtube_google_pool5.h5 \
  --split splits/vasnet_summe_aug_splits.json --split-id 0 --metric summe \
  --save-dir log/summe_augmented_split0
```

For transfer evaluation, use the same four dataset arguments with
`splits/vasnet_summe_tran_splits.json` or
`splits/vasnet_tvsum_tran_splits.json`. Run `--split-id 0` through `4` for the
five random splits reported in the manuscript.

Evaluation-only example:

```bash
python main.py \
  --dataset datasets/eccv16_dataset_summe_google_pool5.h5 \
  --split splits/summe_splits.json --split-id 0 --metric summe \
  --resume log/summe_canonical_split0/summe_iscrl_rl300_split0.pth.tar \
  --evaluate --save-results --save-dir log/summe_canonical_split0
```

## Interpretability

For a qualitative figure, provide the checkpoint, feature HDF5 file, video
key, and a directory containing the corresponding original frames:

```bash
python run_visualization.py \
  --checkpoint log/summe_canonical_split0/summe_iscrl_rl300_split0.pth.tar \
  --dataset datasets/eccv16_dataset_summe_google_pool5.h5 \
  --key video_1 --frame-dir frames/video_1 --top-k 5
```

The script saves the attention matrix, temporal score curve, Grad-CAM++
overlays, and `explanation_scores.npz`. Add `--explain-all` to compute spatial
scores and the combined score for every sampled frame; this is substantially
slower. The raw frames must use the same fixed ImageNet GoogLeNet preprocessing
as the pool5 features for a faithful end-to-end explanation.

`explanation_evaluation.py` implements the paper's Gaussian-noise (`0.01` and
`0.03`), temporal-jitter (`+/-1`), frame-dropout (`5%`), Top-10% overlap, and
high/low/random deletion helpers. Re-evaluate each returned feature sequence
with the same checkpoint to reproduce the stability and deletion protocols.

## Reproducibility notes

- Human annotations are used only for evaluation, never for training or model
  selection.
- The SimCLR projector is optimized during warm-up, frozen during RL, and used
  to produce invariant features for state construction and reward computation.
- The combined explanation score uses separately max-normalized temporal and
  spatial scores with `omega=0.5`; it is not used in the reward or training.
- Checkpoints include model parameters, training arguments, warm-up history,
  optimizer state, and per-video AIM state.
- The numerical values reported in the paper require rerunning all five splits;
  changing software versions, feature files, or random seeds can change results.

## Acknowledgements

The project builds on the problem formulation and evaluation conventions of
[DR-DSN](https://github.com/KaiyangZhou/pytorch-vsumm-reinforce),
[DSR-RL](https://github.com/phaphuang/DSR-RL), and related public video
summarization implementations. Their original licenses and citations should be
consulted when reusing derived material.
