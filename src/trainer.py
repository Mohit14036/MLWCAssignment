"""Generic single-stage trainer used for both ChannelNet stages.

* loss          : MSE between network output and ground-truth (normalised) channel image
* optimiser     : Adam (lr 1e-3, as in the paper)
* LR schedule   : ReduceLROnPlateau on the validation loss
* early stopping: stop after `patience` epochs without validation improvement, keep best weights
* AMP           : optional fp16 autocast + GradScaler on CUDA
* history CSV   : epoch, train_loss, val_loss, lr, seconds  (used for the loss-curve figures)
"""
from __future__ import annotations

import csv
import math
import time

import numpy as np
import torch
import torch.nn as nn


def _to_tensor(a, device):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)


@torch.no_grad()
def predict(model: nn.Module, x, device, batch_size: int = 2048, amp: bool = False) -> np.ndarray:
    """Batched inference; x is a numpy array (N, 1, K, L). Returns numpy float32."""
    model.eval()
    outs = []
    use_amp = amp and device.type == "cuda"
    for i in range(0, len(x), batch_size):
        xb = _to_tensor(x[i:i + batch_size], device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            yb = model(xb)
        outs.append(yb.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # older torch
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_stage(model: nn.Module, x_tr, y_tr, x_va, y_va, stage_cfg: dict, train_cfg: dict,
                device, seed: int, history_csv: str, log=print, name: str = "model") -> dict:
    """Train ``model`` to map x -> y. Returns a summary dict; ``model`` holds the best weights."""
    model.to(device)
    bs = int(train_cfg["batch_size"])
    amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    eval_bs = int(train_cfg.get("eval_batch_size", 2048))

    # keep the whole data set on the device when it fits (72x14 images are tiny)
    xt, yt = _to_tensor(x_tr, device), _to_tensor(y_tr, device)
    xv, yv = _to_tensor(x_va, device), _to_tensor(y_va, device)

    opt = torch.optim.Adam(model.parameters(), lr=float(stage_cfg["lr"]))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=float(stage_cfg["lr_factor"]),
        patience=int(stage_cfg["lr_patience"]), min_lr=float(stage_cfg["min_lr"]))
    scaler = _make_scaler(amp)
    loss_fn = nn.MSELoss()
    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    best, best_state, best_epoch, bad = math.inf, None, 0, 0
    rows = []
    n = xt.shape[0]
    steps = math.ceil(n / bs)
    log(f"[{name}] train {n} images / val {xv.shape[0]} images, {steps} it/epoch, amp={amp}")
    for epoch in range(1, int(stage_cfg["max_epochs"]) + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen).to(device)
        run = 0.0
        for i in range(steps):
            idx = perm[i * bs:(i + 1) * bs]
            xb, yb = xt[idx], yt[idx]
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                loss = loss_fn(model(xb).float(), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item() * xb.shape[0]
        train_loss = run / n

        model.eval()
        with torch.no_grad():
            vs = 0.0
            for i in range(0, xv.shape[0], eval_bs):
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                    vs += loss_fn(model(xv[i:i + eval_bs]).float(), yv[i:i + eval_bs]).item() * \
                        xv[i:i + eval_bs].shape[0]
            val_loss = vs / xv.shape[0]
        sched.step(val_loss)
        lr = opt.param_groups[0]["lr"]
        dt = time.time() - t0
        rows.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr, "seconds": dt})

        improved = val_loss < best * (1 - 1e-4)
        if improved:
            best, best_epoch, bad = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"[{name}] epoch {epoch:3d}  train {train_loss:.4e}  val {val_loss:.4e}  lr {lr:.1e}  "
            f"{dt:5.1f}s {'*' if improved else ''}")
        if bad >= int(stage_cfg["patience"]):
            log(f"[{name}] early stopping (no improvement for {bad} epochs)")
            break

    with open(history_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    model.load_state_dict(best_state)
    return {"best_val_loss": best, "best_epoch": best_epoch, "epochs_run": len(rows),
            "train_seconds": float(sum(r["seconds"] for r in rows))}
