"""Evaluation of conventional estimators and trained ChannelNet models (MSE vs SNR / vs #pilots).

Every method at a given (channel, pilots, SNR) is evaluated on *exactly the same* noisy LS
observations of the same test channels (noise RNG derived from the master seed), so the curves
are directly comparable.

CSV outputs (results/metrics):
    conventional_<ch>_p<P>.csv       snr, LS-linear, LS-cubic, LS-rbf, MMSE-robust, MMSE-estimated, MMSE-ideal
    channelnet_<ch>_p<P>.csv         snr, SR[snr12], ChannelNet[snr12], SR[snr22], ...
    mse_vs_snr_<ch>_p<P>.csv         merge of the two
    mse_vs_pilots_<ch>_snr<S>.csv    pilots, <methods>   at the fixed pilot-sweep SNR
"""
from __future__ import annotations

import csv
import json
import os

import numpy as np
import torch

from .config import iter_experiments
from .dataset import complex_to_images, get_split, images_to_complex, ls_observations
from .estimators import ConventionalEstimators, apply_linear, interpolation_matrix, mse
from .models import build_dncnn, build_srcnn
from .ofdm import OFDMGrid, pilot_mask
from .trainer import predict
from .utils import ckpt_dir, make_rng, snr_tag


# ============================================================================ csv helpers
def write_rows(path: str, rows: list[dict]) -> None:
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def read_rows(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for r in rows:
        out.append({k: (float(v) if v not in ("", None) else np.nan) for k, v in r.items()})
    return out


def metrics_path(cfg, name):
    return os.path.join(cfg["paths"]["results_dir"], "metrics", name)


# ============================================================================ conventional
def evaluate_conventional(cfg: dict, channel: str, n_pilots: int, snrs, log=print) -> list[dict]:
    grid = OFDMGrid.from_config(cfg)
    mask = pilot_mask(cfg, grid, n_pilots)
    H_tr = get_split(cfg, channel, "train", log)
    H_te = get_split(cfg, channel, "test", log)
    bank = ConventionalEstimators(cfg, mask, H_tr,
                                  lambda s: make_rng(cfg, "estcov", channel, n_pilots, str(s)))
    rows = []
    for snr in snrs:
        h_ls = ls_observations(cfg, H_te, channel, "test", n_pilots, snr)
        est = bank.estimate_all(h_ls, snr)
        row = {"snr": snr}
        row.update({k: mse(v, H_te) for k, v in est.items()})
        rows.append(row)
        log(f"  {channel} P={n_pilots:2d} SNR={snr:4.1f} dB  " +
            "  ".join(f"{k}={v:.3e}" for k, v in row.items() if k != "snr"))
    write_rows(metrics_path(cfg, f"conventional_{channel}_p{n_pilots}.csv"), rows)
    return rows


# ============================================================================ ChannelNet
def load_trained(cfg: dict, channel: str, n_pilots: int, snr, device):
    """Load SR + IR weights of one experiment; returns (sr, ir, meta) or None if not trained."""
    d = ckpt_dir(cfg, channel, n_pilots, snr)
    sr_p, ir_p, meta_p = (os.path.join(d, f) for f in ("sr.pt", "ir.pt", "meta.json"))
    if not (os.path.exists(sr_p) and os.path.exists(ir_p) and os.path.exists(meta_p)):
        return None
    with open(meta_p, encoding="utf-8") as f:
        meta = json.load(f)
    sr, ir = build_srcnn(cfg), build_dncnn(cfg)
    sr.load_state_dict(torch.load(sr_p, map_location=device, weights_only=True)["state_dict"])
    ir.load_state_dict(torch.load(ir_p, map_location=device, weights_only=True)["state_dict"])
    return sr.to(device).eval(), ir.to(device).eval(), meta


def channelnet_estimate(cfg, sr, ir, meta, W_in, h_ls, device):
    """LS pilots -> interpolated low-res image -> SRCNN -> DnCNN.  Returns (H_sr, H_cn) complex grids."""
    grid = OFDMGrid.from_config(cfg)
    scale = float(meta["scale"])
    x = complex_to_images(apply_linear(W_in, h_ls, grid.shape)) / scale
    bs = int(cfg["training"].get("eval_batch_size", 2048))
    amp = bool(cfg["training"].get("amp", False))
    s = predict(sr, x, device, bs, amp)
    c = predict(ir, s, device, bs, amp)
    return images_to_complex(s * scale), images_to_complex(c * scale)


def evaluate_channelnet(cfg: dict, channel: str, n_pilots: int, snrs, device, log=print) -> list[dict]:
    grid = OFDMGrid.from_config(cfg)
    mask = pilot_mask(cfg, grid, n_pilots)
    H_te = get_split(cfg, channel, "test", log)
    models = {}
    for ch, p, s in iter_experiments(cfg, channel=channel, pilots=n_pilots):
        m = load_trained(cfg, ch, p, s, device)
        if m is None:
            log(f"  [skip] no checkpoint for {ch} P={p} {snr_tag(s)}")
            continue
        models[snr_tag(s)] = m
    if not models:
        return []
    W_cache = {}
    rows = []
    example = {}
    for snr in snrs:
        h_ls = ls_observations(cfg, H_te, channel, "test", n_pilots, snr)
        row = {"snr": snr}
        for tag, (sr, ir, meta) in models.items():
            interp = meta.get("sr_input_interp", cfg["estimators"]["sr_input_interp"])
            if interp not in W_cache:
                W_cache[interp] = interpolation_matrix(mask, interp)
            H_sr, H_cn = channelnet_estimate(cfg, sr, ir, meta, W_cache[interp], h_ls, device)
            row[f"SR[{tag}]"] = mse(H_sr, H_te)
            row[f"ChannelNet[{tag}]"] = mse(H_cn, H_te)
            if float(snr) == float(cfg["plots"]["example_snr"]) and tag == snr_tag(cfg["plots"]["example_snr"]):
                example = {"truth": H_te[0], "ls_input": apply_linear(W_cache[interp], h_ls[:1], grid.shape)[0],
                           "sr": H_sr[0], "channelnet": H_cn[0], "snr": snr, "model": tag}
        rows.append(row)
        log(f"  {channel} P={n_pilots:2d} SNR={snr:4.1f} dB  " +
            "  ".join(f"{k}={v:.3e}" for k, v in row.items() if k != "snr"))
    write_rows(metrics_path(cfg, f"channelnet_{channel}_p{n_pilots}.csv"), rows)
    if example:
        np.savez(metrics_path(cfg, f"example_{channel}_p{n_pilots}.npz"), **example)
    return rows


# ============================================================================ merge / sweeps
def merge_tables(cfg: dict, channel: str, n_pilots: int) -> list[dict]:
    conv_p = metrics_path(cfg, f"conventional_{channel}_p{n_pilots}.csv")
    dl_p = metrics_path(cfg, f"channelnet_{channel}_p{n_pilots}.csv")
    if not os.path.exists(conv_p):
        return []
    conv = {r["snr"]: r for r in read_rows(conv_p)}
    dl = {r["snr"]: r for r in read_rows(dl_p)} if os.path.exists(dl_p) else {}
    rows = []
    for snr, r in conv.items():
        m = dict(r)
        m.update({k: v for k, v in dl.get(snr, {}).items() if k != "snr"})
        rows.append(m)
    write_rows(metrics_path(cfg, f"mse_vs_snr_{channel}_p{n_pilots}.csv"), rows)
    return rows


def pilot_sweep_table(cfg: dict, channel: str, snr: float) -> list[dict]:
    rows = []
    for p in sorted(cfg["pilots"]["patterns"], reverse=True):
        path = metrics_path(cfg, f"mse_vs_snr_{channel}_p{p}.csv")
        if not os.path.exists(path):
            continue
        for r in read_rows(path):
            if abs(r["snr"] - float(snr)) < 1e-9:
                out = {"pilots": p}
                out.update({k: v for k, v in r.items() if k != "snr"})
                rows.append(out)
    if rows:
        write_rows(metrics_path(cfg, f"mse_vs_pilots_{channel}_snr{int(snr)}.csv"), rows)
    return rows
