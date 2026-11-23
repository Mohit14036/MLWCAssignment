"""Step 4 - TRAINING of ChannelNet (Tasks 3, 4, 5).  Training only; testing is in 05_test.py.

Stage-wise procedure of the paper, for every (channel, #pilots, training SNR) in cfg['experiments']:
  1. LS pilot estimates at the training SNR  ->  interpolated low-resolution image (Gaussian RBF)
  2. SR stage : train SRCNN   LR image -> ground-truth image            (MSE, Adam 1e-3)
  3. IR stage : freeze SRCNN, train DnCNN   SRCNN output -> ground truth  (MSE, Adam 1e-3)
Real and imaginary parts are separate 1-channel images, normalised by a global scale.

    python scripts/04_train.py                                   # all experiments in the config
    python scripts/04_train.py --channel VehA --pilots 48 --snr 22
    python scripts/04_train.py --stage ir --force                # retrain only the DnCNN stage

Existing checkpoints are skipped unless --force is given (makes long runs resumable).
Outputs checkpoints/<ch>/p<P>/snr<S>/{sr.pt, ir.pt, meta.json},
        results/logs/<ch>_p<P>_snr<S>_{sr,ir}_history.csv
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from src.config import base_parser, iter_experiments, load_config  # noqa: E402
from src.dataset import complex_to_images, get_split, image_scale, ls_observations  # noqa: E402
from src.estimators import apply_linear, interpolation_matrix  # noqa: E402
from src.models import build_dncnn, build_srcnn, count_parameters  # noqa: E402
from src.ofdm import OFDMGrid, pilot_mask  # noqa: E402
from src.trainer import predict, train_stage  # noqa: E402
from src.utils import (Logger, ckpt_dir, env_summary, get_device, history_path, json_safe,  # noqa: E402
                       save_json, set_global_seed, snr_tag, stable_int)


def prepare_inputs(cfg, H, channel, split, n_pilots, snr, W_in, scale):
    """Noisy LS pilots -> interpolated LR images (x) and ground-truth images (y), both normalised."""
    grid = OFDMGrid.from_config(cfg)
    h_ls = ls_observations(cfg, H, channel, split, n_pilots, snr)
    x = complex_to_images(apply_linear(W_in, h_ls, grid.shape)) / scale
    y = complex_to_images(H) / scale
    return x, y


def train_one(cfg, channel, n_pilots, snr, stage, force, device, log):
    tag = snr_tag(snr)
    d = ckpt_dir(cfg, channel, n_pilots, snr)
    sr_path, ir_path, meta_path = (os.path.join(d, f) for f in ("sr.pt", "ir.pt", "meta.json"))
    do_sr = stage in ("all", "sr") and (force or not os.path.exists(sr_path))
    do_ir = stage in ("all", "ir") and (force or not os.path.exists(ir_path) or do_sr)
    if not (do_sr or do_ir):
        log(f"[skip] {channel} P={n_pilots} {tag}: checkpoints exist (use --force to retrain)")
        return
    log(f"========== {channel} | {n_pilots} pilots | train {tag} ==========")
    seed = int(cfg["seed"]) + stable_int((channel, n_pilots, tag)) % 100000
    set_global_seed(seed, bool(cfg["training"].get("deterministic", True)))

    grid = OFDMGrid.from_config(cfg)
    mask = pilot_mask(cfg, grid, n_pilots)
    interp = cfg["estimators"]["sr_input_interp"]
    W_in = interpolation_matrix(mask, interp)
    H_tr = get_split(cfg, channel, "train", log)
    H_va = get_split(cfg, channel, "val", log)
    scale = image_scale(H_tr)
    x_tr, y_tr = prepare_inputs(cfg, H_tr, channel, "train", n_pilots, snr, W_in, scale)
    x_va, y_va = prepare_inputs(cfg, H_va, channel, "val", n_pilots, snr, W_in, scale)
    log(f"data: train {x_tr.shape}, val {x_va.shape}, scale {scale:.4f}, LR interp '{interp}'")

    meta = {"channel": channel, "pilots": n_pilots, "train_snr": snr, "tag": tag, "seed": seed,
            "scale": scale, "sr_input_interp": interp, "model": cfg["model"], "training": cfg["training"],
            "dataset": cfg["dataset"], "config_file": cfg["_config_path"], "environment": env_summary()}
    if os.path.exists(meta_path) and not force:
        import json
        with open(meta_path, encoding="utf-8") as f:
            meta = {**json.load(f), **{k: meta[k] for k in ("scale", "environment")}}

    # ---------------- stage 1: super-resolution (SRCNN)
    sr = build_srcnn(cfg)
    if do_sr:
        log(f"SRCNN parameters: {count_parameters(sr)}")
        res = train_stage(sr, x_tr, y_tr, x_va, y_va, cfg["training"]["sr"], cfg["training"], device, seed,
                          history_path(cfg, channel, n_pilots, snr, "sr"), log, name=f"SR {channel}/{n_pilots}/{tag}")
        torch.save({"state_dict": sr.state_dict(), "meta": json_safe(meta)}, sr_path)
        meta["sr_result"] = res
        meta["sr_params"] = count_parameters(sr)
        log(f"saved {sr_path}  ({res})")
    else:
        sr.load_state_dict(torch.load(sr_path, map_location=device, weights_only=True)["state_dict"])
        sr.to(device)

    # ---------------- stage 2: image restoration (DnCNN) on top of the frozen SRCNN
    if do_ir:
        bs = int(cfg["training"].get("eval_batch_size", 2048))
        s_tr = predict(sr, x_tr, device, bs)
        s_va = predict(sr, x_va, device, bs)
        ir = build_dncnn(cfg)
        log(f"DnCNN parameters: {count_parameters(ir)}")
        res = train_stage(ir, s_tr, y_tr, s_va, y_va, cfg["training"]["ir"], cfg["training"], device, seed + 1,
                          history_path(cfg, channel, n_pilots, snr, "ir"), log, name=f"IR {channel}/{n_pilots}/{tag}")
        torch.save({"state_dict": ir.state_dict(), "meta": json_safe(meta)}, ir_path)
        meta["ir_result"] = res
        meta["ir_params"] = count_parameters(ir)
        log(f"saved {ir_path}  ({res})")
    save_json(meta, meta_path)


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--channel", default=None, help="VehA | SUI5 (default: all in config)")
    ap.add_argument("--pilots", type=int, default=None)
    ap.add_argument("--snr", default=None, help="training SNR, e.g. 12, 22 or mix")
    ap.add_argument("--stage", default="all", choices=["all", "sr", "ir"])
    ap.add_argument("--force", action="store_true", help="retrain even if checkpoints exist")
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    log = Logger(os.path.join(cfg["paths"]["results_dir"], "logs", "04_train.log"))
    device = get_device(cfg)
    log("device:", device, env_summary())
    t0 = time.time()
    exps = list(iter_experiments(cfg, args.channel, args.pilots, args.snr))
    log(f"{len(exps)} experiment(s): {exps}")
    budget_h = float(cfg["training"].get("time_budget_hours", 0) or 0)
    for ch, p, s in exps:
        if budget_h > 0 and (time.time() - t0) / 3600 > budget_h:
            log(f"[budget] {budget_h} h training budget used up - skipping {ch} P={p} {snr_tag(s)} "
                f"(re-run this script later to train it)")
            continue
        train_one(cfg, ch, p, s, args.stage, args.force, device, log)
    log(f"total training time {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
