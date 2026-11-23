# ChannelNet - Deep-Learning-Based OFDM Channel Estimation (COM 837, Assignment 3)

Reproduction of **ChannelNet** (M. Soltani *et al.*, "Deep Learning-Based Channel Estimation",
arXiv:1810.05893v4). The 72 x 14 LTE time-frequency channel grid is treated as an image and reconstructed
from sparse, noisy LS pilot estimates by **SRCNN super-resolution followed by DnCNN denoising**. It is
compared with LS + 2-D interpolation and MMSE estimators (ideal MMSE, estimated MMSE, ideal ALMMSE) for the
**VehA** and **SUI5** channel models.

## 1. Contents

| item | location |
|---|---|
| report (PDF) and its LaTeX source | `report/IMT2023528_A3_Report.pdf`, `report/main.tex`, `report/Images/` |
| library code (OFDM, channels, estimators, networks, training, evaluation, plots) | `src/` |
| pipeline scripts (separate **training** and **testing** scripts) | `scripts/01 ... 06` |
| run everything in order | `run_all.py` |
| configuration of the submitted run / base settings / Vienna cross-check | `configs/final.yaml`, `configs/default.yaml`, `configs/vienna_eval.yaml` |
| Vienna 5G simulator channel generation (MATLAB) | `matlab/generate_vienna_channels.m` (+ `README_vienna.md`) |
| trained models (14) | `checkpoints/<channel>/p<pilots>/snr<S>/{sr.pt, ir.pt, meta.json}` |
| results: figures / MSE tables / training histories + logs | `results/figures/`, `results/metrics/`, `results/logs/` |

## 2. Task -> code map

| Task | Where |
|---|---|
| 1 OFDM grid, channels, lattice pilots, LS observations, Fig. 2 | `src/ofdm.py`, `src/channel_models.py`, `src/dataset.py`, `matlab/generate_vienna_channels.m`, `scripts/01_generate_data.py`, `scripts/02_visualize_data.py` |
| 2 LS + 2-D interpolation, ideal / estimated MMSE, ideal ALMMSE | `src/estimators.py`, `scripts/03_conventional.py` |
| 3 SR stage (SRCNN 9-1-5) | `src/models.py` (`SRCNN`), `scripts/04_train.py` |
| 4 IR stage (20-layer residual DnCNN), cascade = ChannelNet | `src/models.py` (`DnCNN`, `ChannelNet`), `scripts/04_train.py` |
| 5 Training: 80/10/10 split, SNR-specific (12 / 22 dB) + multi-SNR, seeds, checkpoints | `src/trainer.py`, `scripts/04_train.py`, `configs/final.yaml` |
| 6 Evaluation: MSE vs SNR, VehA vs SUI5, MSE vs number of pilots | `src/evaluation.py`, `scripts/05_test.py`, `scripts/06_make_plots.py` |

## 3. Setup

Python >= 3.10 (tested with 3.12/3.13), PyTorch 2.x, NumPy 2.x, SciPy.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate        Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt          # for a CUDA build of torch see https://pytorch.org
```

## 4. How to run

All commands are run from the top-level folder of the submission. Every step is seeded (master seed
2026) and idempotent.

**a) Test the submitted models (no training).** Regenerates the data, evaluates the baselines and the 14
saved checkpoints, and redraws all figures into `results/`:

```bash
python run_all.py --config configs/final.yaml --steps data viz conventional test plots
```

**b) Full run including training** (GPU recommended; the submitted models were trained on a Kaggle
Tesla T4 in 424 min). Training skips models whose checkpoint already exists - delete `checkpoints/` (or use
`--force`) to retrain:

```bash
python run_all.py --config configs/final.yaml
```

**c) Individual steps:**

```bash
python scripts/01_generate_data.py  --config configs/final.yaml   # channel data sets + statistics
python scripts/02_visualize_data.py --config configs/final.yaml   # Fig. 2, pilot patterns
python scripts/03_conventional.py   --config configs/final.yaml   # LS / MMSE / ALMMSE baselines
python scripts/04_train.py          --config configs/final.yaml   # TRAINING only (SR stage, then IR stage)
python scripts/05_test.py           --config configs/final.yaml   # TESTING only (loads checkpoints)
python scripts/06_make_plots.py     --config configs/final.yaml   # all result figures

python scripts/04_train.py --config configs/final.yaml --channel VehA --pilots 48 --snr 22   # one model
```

**d) Vienna-simulator channels (optional, MATLAB).** `matlab/generate_vienna_channels.m` generates the
VehA/SUI5 channels with the course's Vienna 5G Link-Level Simulator (see `matlab/README_vienna.md`;
writes `data/vienna_*.mat`). Then

```bash
python run_all.py --config configs/vienna_eval.yaml --steps data conventional test plots
```

tests the trained networks on these channels (report Section 6.4; the resulting tables are also in
`results/metrics/vienna_mse_vs_snr_*.csv`).

## 5. Results

| Required plot | File in `results/figures/` |
|---|---|
| Ground-truth real/imaginary channel images (paper Fig. 2) | `fig2_ground_truth_VehA.png`, `fig2_ground_truth_SUI5.png` |
| Training/validation loss, SR and IR stages | `loss_curves/loss_<ch>_p<P>_snr<S>.png` (all 14 models) |
| MSE vs SNR, VehA (paper Fig. 3) / SUI5 (paper Fig. 4), 48 pilots | `fig3_VehA_mse_vs_snr.png`, `fig4_SUI5_mse_vs_snr.png` |
| Deep high-SNR case, VehA and SUI5 separately | `high_snr_VehA_vs_SUI5.png` |
| MSE vs number of pilots at fixed SNR (paper Fig. 5) | `fig5_mse_vs_pilots_snr22.png` |
| additional | `pilot_patterns.png`, `lowpilots_VehA_mse_vs_snr.png`, `snr_generalization.png`, `example_reconstruction_*.png`, `lr_input_example_*.png`, `vienna_*.png` |

Numbers: `results/metrics/mse_vs_snr_<ch>_p<P>.csv`, `mse_vs_pilots_<ch>_snr22.csv`,
`channel_statistics.json`. Training histories: `results/logs/*_history.csv`. Each checkpoint's `meta.json`
stores the configuration, seed, normalisation, best epochs/losses and the environment.

## 6. Key settings

* LTE 1.4 MHz grid: 72 sub-carriers x 14 OFDM symbols, 15 kHz, normal CP; `Y = H X + Z`, SNR = 1/sigma^2,
  `H_LS = Y_p / X_p`.
* VehA (ITU-R M.1225) and SUI5 (0/4/10 us, 0/-5/-10 dB), 2.1 GHz, 50 km/h (f_D = 97 Hz), Jakes fading,
  40 000 frames per model (32 000 / 4 000 / 4 000).
* Lattice pilots: 48 (LTE CRS diamond), plus 36, 24, 16 and 8.
* ChannelNet: Gaussian-RBF interpolated LS input -> SRCNN (9x9x64, 1x1x32, 5x5x1) -> DnCNN (20 layers,
  64 filters, BN, residual). Two-stage MSE training, Adam (lr 1e-3), batch 128, ReduceLROnPlateau, early
  stopping.
* Training SNRs 12 dB (deep low-SNR) and 22 dB (deep high-SNR), a multi-SNR model (U[0, 30] dB), and
  22 dB models for 24/16/8 pilots.
