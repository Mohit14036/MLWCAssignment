"""Dataset preparation.

* ground-truth channels H (N, 72, 14) are generated once per channel model and cached
  (or loaded from a Vienna-LTE ``.mat`` export),
* split 80 / 10 / 10 into train / validation / test,
* noisy LS pilot observations are created deterministically from (seed, channel, split, pilots, SNR),
* complex grids <-> two real 2-D images (real part, imaginary part), normalised by a global scale.
"""
from __future__ import annotations

import os

import numpy as np

from .channel_models import generate_tdl_channels
from .ofdm import OFDMGrid, ls_pilot_observation, pilot_mask
from .utils import make_rng

SPLITS = ("train", "val", "test")


# ============================================================================ channels
def channel_cache_file(cfg: dict, channel: str) -> str:
    n = int(cfg["dataset"]["n_realizations"])
    src = cfg["channel"].get("source", "python")
    return os.path.join(cfg["paths"]["data_dir"], f"channels_{channel}_{src}_N{n}_seed{cfg['seed']}.npy")


def _load_mat_channels(cfg: dict, channel: str, grid: OFDMGrid) -> np.ndarray:
    """Load channels exported from the Vienna LTE simulator (see matlab/README_vienna.md)."""
    from scipy.io import loadmat

    from .config import PROJECT_ROOT
    files = cfg["channel"]["mat_files"][channel]
    files = [files] if isinstance(files, str) else list(files)
    parts = []
    for path in files:
        if not os.path.isabs(path):
            path = os.path.join(PROJECT_ROOT, path)
        parts.append(_load_one_mat(path, cfg["channel"].get("mat_variable", "H"), grid))
    H = np.concatenate(parts, axis=0)
    # normalise to unit average power so that SNR definitions match
    H = H / np.sqrt(np.mean(np.abs(H) ** 2))
    return H.astype(np.complex64)


def _load_one_mat(path: str, var: str, grid: OFDMGrid) -> np.ndarray:
    from scipy.io import loadmat
    try:
        H = loadmat(path)[var]
    except NotImplementedError:          # MATLAB v7.3 (HDF5) file
        import h5py
        with h5py.File(path, "r") as f:
            raw = f[var][()]
        H = raw["real"] + 1j * raw["imag"] if raw.dtype.names else raw
        H = np.transpose(H)              # h5py reverses MATLAB's dimension order
    H = np.asarray(H)
    K, L = grid.shape
    if H.shape[1:] == (K, L):
        pass
    elif H.shape[:2] == (K, L):
        H = np.moveaxis(H, -1, 0)
    else:
        raise ValueError(f"cannot interpret Vienna channel array of shape {H.shape}")
    return H


def load_channels(cfg: dict, channel: str, log=print) -> np.ndarray:
    """Return all channel realisations of one model, generating & caching them if needed."""
    path = channel_cache_file(cfg, channel)
    if os.path.exists(path):
        return np.load(path)
    grid = OFDMGrid.from_config(cfg)
    src = cfg["channel"].get("source", "python")
    if src == "python":
        n = int(cfg["dataset"]["n_realizations"])
        log(f"generating {n} {channel} channel realisations ...")
        H = generate_tdl_channels(channel, n, grid, cfg["channel"], make_rng(cfg, "channel", channel))
    elif src == "mat":
        log(f"loading Vienna channels for {channel} ...")
        H = _load_mat_channels(cfg, channel, grid)
    else:
        raise ValueError(f"unknown channel source {src}")
    np.save(path, H)
    log(f"saved {H.shape} -> {path}")
    return H


def split_channels(cfg: dict, H: np.ndarray) -> dict:
    """Contiguous 80/10/10 split (realisations are i.i.d., so no shuffling is needed)."""
    fr = cfg["dataset"]["split"]
    n = H.shape[0]
    n_tr = int(round(fr[0] * n))
    n_va = int(round(fr[1] * n))
    return {"train": H[:n_tr], "val": H[n_tr:n_tr + n_va], "test": H[n_tr + n_va:]}


def get_split(cfg: dict, channel: str, split: str, log=print) -> np.ndarray:
    H = split_channels(cfg, load_channels(cfg, channel, log))[split]
    if split == "test":
        H = H[: int(cfg["dataset"].get("n_test_eval", len(H)))]
    return H


# ============================================================================ observations
def draw_snr(cfg: dict, snr, n: int, rng: np.random.Generator):
    """Fixed SNR -> scalar;  'mix' -> per-frame SNR ~ U[mix_range] (multi-SNR training)."""
    if isinstance(snr, str):
        if snr != "mix":
            raise ValueError(f"unknown SNR spec {snr}")
        lo, hi = cfg["snr"]["mix_range"]
        return rng.uniform(lo, hi, size=n)
    return float(snr)


def ls_observations(cfg: dict, H: np.ndarray, channel: str, split: str, n_pilots: int, snr) -> np.ndarray:
    """Deterministic noisy LS pilot estimates (N, P) for one (channel, split, pilots, SNR)."""
    grid = OFDMGrid.from_config(cfg)
    mask = pilot_mask(cfg, grid, n_pilots)
    rng = make_rng(cfg, "noise", channel, split, n_pilots, str(snr))
    snr_val = draw_snr(cfg, snr, H.shape[0], rng)
    return ls_pilot_observation(H, mask, snr_val, rng)


# ============================================================================ images
def complex_to_images(Hc: np.ndarray) -> np.ndarray:
    """(N, K, L) complex -> (2N, 1, K, L) float32: first N real-part images, then N imaginary-part.

    As in the paper, real and imaginary parts are treated as two independent 2-D images."""
    return np.concatenate([Hc.real, Hc.imag], axis=0)[:, None].astype(np.float32)


def images_to_complex(img: np.ndarray) -> np.ndarray:
    """Inverse of complex_to_images."""
    img = np.asarray(img)[:, 0]
    n = img.shape[0] // 2
    return (img[:n] + 1j * img[n:]).astype(np.complex64)


def image_scale(H_train: np.ndarray) -> float:
    """Global normalisation constant = std of the real/imag entries of the training targets."""
    return float(np.sqrt(np.mean(np.abs(H_train) ** 2) / 2.0))
