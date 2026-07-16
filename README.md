# Hypernetworks for Sound Event Detection — URBAN-SED reproduction

A clean, reproducible [PyTorch Lightning](https://lightning.ai/) +
[Hydra](https://hydra.cc/) reimplementation of the URBAN-SED experiments from:

> S. Singh, H. Phan, E. Benetos, **"Hypernetworks for Sound Event Detection: A
> Proof-of-Concept."**

The paper relaxes the weight sharing of the recurrent block of a CRNN using a
**dynamic hypernetwork** (HyperLSTM) and shows a ≈3–4% segment-wise F1
improvement over CRNN baselines on URBAN-SED and TUT-SED Synthetic 2016. This
repository reproduces **Table 2 (URBAN-SED)** and adds a **multi-seed** study.

## Models

| Config | Recurrent block | Notes |
|---|---|---|
| `crnn_bi`   | BiLSTM, hidden 512   | Bi-baseline |
| `crnn_uni`  | LSTM, hidden 256     | Uni-baseline |
| `hcrnn32`   | HyperLSTM, main 256 / hyper 32 / z=8   | HCRNN-32 |
| `hcrnn64`   | HyperLSTM, main 256 / hyper 64 / z=8   | HCRNN-64 |
| `hcrnn128`  | HyperLSTM, main 256 / hyper 128 / z=8  | HCRNN-128 |
| `hcrnn256`  | HyperLSTM, main 256 / hyper 256 / z=8  | HCRNN-256 |

The paper reports HCRNN-64 and HCRNN-128; `hcrnn32` and `hcrnn256` extend the
hypernetwork hidden-size sweep.

All models share the same CNN front-end: 4 blocks of `3×3 conv → BatchNorm →
ReLU → 2×2 pool` with channels `(64, 128, 256, 512)`, followed by the recurrent
block and a time-distributed linear + sigmoid classifier over the 10 URBAN-SED
classes. See `src/modules/components/{crnn,hyperlstm}.py`.

## Project layout

```
config/                 Hydra configs
  train.yaml            main training config (defaults list)
  eval.yaml             standalone evaluation config
  experiment/           crnn_bi, crnn_uni, hcrnn128, hcrnn64
  model/crnn.yaml       CRNN/HCRNN network definition
  modules/sed.yaml      LightningModule (optimizer, net)
  data/urban_sed.yaml   datamodule
  callbacks|trainer|logger|paths|hydra/
src/
  preprocess.py         log-mel feature extraction + strong labels
  train.py              training + best-10-checkpoint test evaluation
  eval.py               standalone checkpoint evaluation
  aggregate_results.py  build Table-2 (mean ± std over seeds)
  data/                 datamodule + chunk dataset
  modules/              LightningModule + CRNN/HyperLSTM
  utils/                sed_eval segment-F1, file-level evaluation, helpers
scripts/
  download_urbansed.sh  download + extract URBAN-SED v2.0.0
  run_seeds.sh          run all models × seeds across GPUs
```

## Setup

```bash
conda create -y -n hsed python=3.11
conda activate hsed
pip install -r requirements.txt   # installs a CUDA torch build by default
```

## Data

Download URBAN-SED v2.0.0 (~6.5 GB, 10,000 10-second soundscapes with strong
annotations):

```bash
scripts/download_urbansed.sh data/URBAN-SED
# -> data/URBAN-SED/URBAN-SED_v2.0.0/{audio,annotations}/{train,validate,test}
```

## Feature extraction

96-band log-mel (22.05 kHz, `n_fft=1024`, `hop=256`, 50–11025 Hz), z-score
normalised with **training-set** statistics, chunked into 2 s windows with a
0.5 s hop (paper Section 3.3). Features are cached once and reused by every run:

```bash
python -m src.preprocess \
  --dataset_root data/URBAN-SED/URBAN-SED_v2.0.0 \
  --out_dir data/urban_sed_features \
  --num_workers 16
# writes features/, labels/, manifest.json, stats.json
```

## Training

Single run:

```bash
python -m src.train experiment=hcrnn128 seed=42
```

Training uses Adam (`lr=4e-5`), batch size 64, up to 250 epochs with early
stopping (patience 10), monitoring validation loss (paper Section 3.4). Outputs
(checkpoints, `csv/` logs, `results.json`) are written under
`logs/<experiment>/seed_<seed>/<timestamp>/`.

### Evaluation protocol

After fitting, the model is scored on the **test set across its best-10
checkpoints** (by validation loss) and the **mean** segment-wise F1 is reported,
matching the paper ("evaluated on the test sets across 10 epochs and the mean of
the results across the epochs was reported"). Segment-wise F1 uses `sed_eval`
with 1-second segments.

## Multi-seed study (Table 2)

Run all four models across several seeds, one job per GPU:

```bash
# seeds, experiments, GPU ids
scripts/run_seeds.sh "42 1337 2024" "crnn_bi crnn_uni hcrnn32 hcrnn64 hcrnn128 hcrnn256" "0 1 2 3"
```

Aggregate into per-seed and mean ± std tables:

```bash
python -m src.aggregate_results --logs_dir logs --out_dir results
# -> results/table2.csv, results/table2.md
```

## Parameter count and walltime

Measured with `scripts/params_walltime.py` (single NVIDIA L40S, PyTorch 2.5.1,
float32, no JIT/torch.compile, batch 64 x 2 s chunks of 172x96 log-mel, Adam +
BCE; train step = forward+backward+optimizer, inference under `no_grad`):

| Model | Trainable params (M) | Train step (ms) | Inference (ms/clip) |
|---|---|---|---|
| crnn_uni  | 2.34 | 65.2  | 0.31 |
| crnn_bi   | 5.76 | 69.2  | 0.33 |
| hcrnn32   | 2.47 | 483.2 | 1.48 |
| hcrnn64   | 2.59 | 482.6 | 1.53 |
| hcrnn128  | 2.84 | 505.7 | 1.53 |
| hcrnn256  | 3.44 | 504.7 | 1.53 |

The HCRNNs have fewer parameters than the Bi-baseline but are ~7x slower per
step: the HyperLSTM is a per-timestep Python loop, whereas `nn.LSTM` uses the
fused cuDNN kernel. Parameter count therefore does not imply runtime speed.
Raw numbers and environment details: `results/params_walltime.{md,csv}`.

## Standalone evaluation

```bash
python -m src.eval experiment=hcrnn128 \
  ckpt_path=logs/hcrnn128/seed_42/<ts>/checkpoints/last.ckpt split=test
```

## Reproduction notes

These choices make explicit where the paper underspecifies implementation
details; all are configurable:

- **CNN pooling.** The paper states 2×2 pooling and a 512-dimensional recurrent
  input. To keep the per-frame time resolution required for strong-label SED,
  pooling is applied along frequency (`96 → 6`) with the time axis preserved, and
  a final adaptive average pool collapses frequency to 1 (→ 512-d). Adjust via
  `model.time_pool` / `model.freq_pool`.
- **HyperLSTM.** Implements the dynamic hypernetwork of Ha et al. (2017) with an
  intermediate scaling vector `d(z)` (paper Eq. 15), main hidden 256, hyper
  hidden 64/128, embedding size `z = 8`.
- **Reference events** for scoring are derived from the strong-label frame
  targets; estimated events are thresholded model probabilities
  (`eval_threshold`, optional `median_filter_frames`).
- **Loss / target.** Frame-level binary cross-entropy over the 10 classes.

## Citation

```bibtex
@inproceedings{singh_hypernetworks_sed,
  title     = {Hypernetworks for Sound Event Detection: A Proof-of-Concept},
  author    = {Singh, Shubhr and Phan, Huy and Benetos, Emmanouil},
  booktitle = {IEEE ICASSP},
}
@inproceedings{ha2017hypernetworks,
  title     = {HyperNetworks},
  author    = {Ha, David and Dai, Andrew and Le, Quoc V.},
  booktitle = {ICLR},
  year      = {2017}
}
@inproceedings{salamon2017scaper,
  title     = {Scaper: A library for soundscape synthesis and augmentation},
  author    = {Salamon, Justin and MacConnell, Duncan and Cartwright, Mark and Li, Peter and Bello, Juan Pablo},
  booktitle = {IEEE WASPAA},
  year      = {2017}
}
```
