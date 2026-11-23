"""All figures of the report (matplotlib, saved as PNG into results/figures)."""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .evaluation import metrics_path, read_rows  # noqa: E402

# consistent style per method ---------------------------------------------------------------
STYLE = {
    "LS-linear": dict(color="#7f7f7f", marker="s", ls="--", label="LS + linear interp."),
    "LS-cubic": dict(color="#bcbd22", marker="d", ls="--", label="LS + cubic interp."),
    "LS-rbf": dict(color="#8c564b", marker="v", ls="--", label="LS + RBF interp. (SR input)"),
    "MMSE-robust": dict(color="#17becf", marker="x", ls="-.", label="Robust LMMSE (unif. PDP)"),
    "MMSE-estimated": dict(color="#ff7f0e", marker="^", ls="-.", label="Estimated MMSE"),
    "ALMMSE-ideal": dict(color="#e377c2", marker="D", ls="-.", label="Ideal ALMMSE"),
    "MMSE-ideal": dict(color="#000000", marker="o", ls="-", label="Ideal MMSE"),
    "SR[snr12]": dict(color="#2ca02c", marker=None, ls=":", label="SR only (12 dB)"),
    "SR[snr22]": dict(color="#d62728", marker=None, ls=":", label="SR only (22 dB)"),
    "SR[snrmix]": dict(color="#9467bd", marker=None, ls=":", label="SR only (0-30 dB mix)"),
    "SR[snr20]": dict(color="#1f77b4", marker=None, ls=":", label="SR only (20 dB)"),
    "ChannelNet[snr12]": dict(color="#2ca02c", marker="*", ls="-", label="Deep Low-SNR (ChannelNet @ 12 dB)"),
    "ChannelNet[snr22]": dict(color="#d62728", marker="P", ls="-", label="Deep High-SNR (ChannelNet @ 22 dB)"),
    "ChannelNet[snr20]": dict(color="#1f77b4", marker="*", ls="-", label="ChannelNet (trained 20 dB)"),
    "ChannelNet[snrmix]": dict(color="#9467bd", marker="h", ls="-", label="ChannelNet (multi-SNR 0-30 dB)"),
}
# paper Figs. 3/4: deep low-SNR, deep high-SNR, ideal MMSE, estimated MMSE, ideal ALMMSE (+ LS, SR-only, mix)
MAIN_METHODS = ["LS-linear", "LS-rbf", "MMSE-estimated", "ALMMSE-ideal", "MMSE-ideal",
                "SR[snr12]", "ChannelNet[snr12]", "SR[snr22]", "ChannelNet[snr22]", "ChannelNet[snrmix]"]


def _style(name):
    return STYLE.get(name, dict(marker="o", ls="-", label=name))


def _save(fig, cfg, name):
    path = os.path.join(cfg["paths"]["results_dir"], "figures", name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=int(cfg["plots"].get("dpi", 150)), bbox_inches="tight")
    plt.close(fig)
    return path


# ============================================================================ data figures
def plot_ground_truth(cfg, H: np.ndarray, channel: str):
    """Fig. 2: real and imaginary part of one channel realisation as 72 x 14 images."""
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 5.2))
    for ax, part, title in zip(axs, (H.real, H.imag), ("(a) real part", "(b) imaginary part")):
        im = ax.imshow(part, aspect="auto", origin="lower", cmap="viridis",
                       extent=[0.5, H.shape[1] + 0.5, 0.5, H.shape[0] + 0.5])
        ax.set_xlabel("OFDM symbol (time)")
        ax.set_ylabel("sub-carrier (frequency)")
        ax.set_title(f"{channel}: {title}")
        fig.colorbar(im, ax=ax, fraction=0.05)
    fig.suptitle(f"Ground-truth channel time-frequency response ({channel}, 72x14)")
    fig.tight_layout()
    return _save(fig, cfg, f"fig2_ground_truth_{channel}.png")


def plot_pilot_patterns(cfg, masks: dict):
    n = len(masks)
    fig, axs = plt.subplots(1, n, figsize=(2.3 * n, 5), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (p, m) in zip(axs, sorted(masks.items(), reverse=True)):
        ax.imshow(m, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1.3,
                  extent=[0.5, m.shape[1] + 0.5, 0.5, m.shape[0] + 0.5])
        ax.set_title(f"{p} pilots")
        ax.set_xlabel("OFDM symbol")
    axs[0].set_ylabel("sub-carrier")
    fig.suptitle("Lattice (diamond) pilot patterns on the 72x14 grid (black = pilot)")
    fig.tight_layout()
    return _save(fig, cfg, "pilot_patterns.png")


def plot_low_res_example(cfg, truth, ls_interp, mask, channel, snr, n_pilots):
    """Ground truth vs sparse noisy LS pilots vs interpolated low-resolution input image."""
    sparse = np.full(truth.shape, np.nan)
    sparse[mask] = ls_interp[mask].real
    fig, axs = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    vmin, vmax = truth.real.min(), truth.real.max()
    for ax, img, t in zip(axs, (truth.real, sparse, ls_interp.real),
                          ("ground truth (real)", f"noisy LS at {n_pilots} pilots", "interpolated LR input")):
        im = ax.imshow(img, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis",
                       extent=[0.5, truth.shape[1] + 0.5, 0.5, truth.shape[0] + 0.5])
        ax.set_title(t)
        ax.set_xlabel("OFDM symbol")
    axs[0].set_ylabel("sub-carrier")
    fig.colorbar(im, ax=axs, fraction=0.03)
    fig.suptitle(f"{channel}, SNR = {snr} dB: input of the SR network")
    return _save(fig, cfg, f"lr_input_example_{channel}_p{n_pilots}.png")


# ============================================================================ training
def plot_loss_curves(cfg, channel, pilots, tag, hist_sr: list, hist_ir: list):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, hist, title in zip(axs, (hist_sr, hist_ir), ("SR stage (SRCNN)", "IR stage (DnCNN)")):
        if not hist:
            ax.set_visible(False)
            continue
        ep = [h["epoch"] for h in hist]
        ax.semilogy(ep, [h["train_loss"] for h in hist], "-", label="training")
        ax.semilogy(ep, [h["val_loss"] for h in hist], "--", label="validation")
        best = int(np.argmin([h["val_loss"] for h in hist]))
        ax.axvline(ep[best], color="grey", lw=0.8, ls=":", label=f"best epoch {int(ep[best])}")
        ax.set_xlabel("epoch")
        ax.set_ylabel("MSE loss (normalised images)")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.suptitle(f"Loss curves - {channel}, {pilots} pilots, {tag}")
    fig.tight_layout()
    return _save(fig, cfg, f"loss_curves/loss_{channel}_p{pilots}_{tag}.png")


# ============================================================================ MSE curves
def _mse_axes(ax, rows, methods, title):
    snr = np.array([r["snr"] for r in rows])
    for m in methods:
        if m not in rows[0]:
            continue
        y = np.array([r[m] for r in rows])
        st = _style(m)
        ax.semilogy(snr, y, color=st.get("color"), marker=st.get("marker"), ls=st.get("ls", "-"),
                    label=st.get("label", m), ms=5, lw=1.4)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)


def _rows(cfg, channel, pilots):
    p = metrics_path(cfg, f"mse_vs_snr_{channel}_p{pilots}.csv")
    return read_rows(p) if os.path.exists(p) else []


def plot_mse_vs_snr_figure(cfg, fig_name, channel, pilots_list, methods=MAIN_METHODS):
    pilots_list = pilots_list if isinstance(pilots_list, (list, tuple)) else [pilots_list]
    avail = [(p, _rows(cfg, channel, p)) for p in pilots_list]
    avail = [(p, r) for p, r in avail if r]
    if not avail:
        return None
    fig, axs = plt.subplots(1, len(avail), figsize=(6.8 * len(avail), 5), squeeze=False)
    for ax, (p, rows) in zip(axs[0], avail):
        _mse_axes(ax, rows, methods, f"{channel}, {p} pilots")
    axs[0][-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return _save(fig, cfg, f"{fig_name}_{channel}_mse_vs_snr.png")


def plot_high_snr_comparison(cfg, pilots):
    """Separate VehA / SUI5 MSE-vs-SNR panels for the deep high-SNR (22 dB-trained) model."""
    methods = ["LS-linear", "LS-rbf", "MMSE-estimated", "ALMMSE-ideal", "MMSE-ideal", "SR[snr22]",
               "ChannelNet[snr22]"]
    chans = [c for c in cfg["channel"]["models"] if _rows(cfg, c, pilots)]
    if not chans:
        return None
    fig, axs = plt.subplots(1, len(chans), figsize=(6.8 * len(chans), 5), squeeze=False)
    for ax, c in zip(axs[0], chans):
        _mse_axes(ax, _rows(cfg, c, pilots), methods, f"{c}, {pilots} pilots - deep high-SNR (22 dB) model")
        ax.axvline(22, color="grey", ls=":", lw=1)
    axs[0][-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    return _save(fig, cfg, "high_snr_VehA_vs_SUI5.png")


def plot_generalization(cfg, pilots):
    """Gap (dB) of every ChannelNet model to the ideal MMSE versus SNR - shows SNR mismatch."""
    chans = [c for c in cfg["channel"]["models"] if _rows(cfg, c, pilots)]
    if not chans:
        return None
    fig, axs = plt.subplots(1, len(chans), figsize=(6.8 * len(chans), 4.5), squeeze=False)
    for ax, c in zip(axs[0], chans):
        rows = _rows(cfg, c, pilots)
        snr = np.array([r["snr"] for r in rows])
        ref = np.array([r["MMSE-ideal"] for r in rows])
        for m in [k for k in rows[0] if k.startswith("ChannelNet") or k in ("LS-linear", "MMSE-estimated")]:
            st = _style(m)
            ax.plot(snr, 10 * np.log10(np.array([r[m] for r in rows]) / ref), color=st.get("color"),
                    marker=st.get("marker"), ls=st.get("ls", "-"), label=st.get("label", m))
        for s in (12, 22):
            ax.axvline(s, color="grey", ls=":", lw=1)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("MSE / MSE(ideal MMSE)  [dB]")
        ax.set_title(f"{c}, {pilots} pilots: SNR generalisation")
        ax.grid(True, alpha=0.3)
    axs[0][-1].legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, cfg, "snr_generalization.png")


def plot_pilot_sweep(cfg, snr):
    """Paper Fig. 5: MSE vs number of pilots at a fixed SNR (ChannelNet trained at that SNR)."""
    chans = []
    first = cfg["plots"].get("fig5", {}).get("channel")
    order = [first] + [c for c in cfg["channel"]["models"] if c != first] if first else cfg["channel"]["models"]
    for c in order:
        p = metrics_path(cfg, f"mse_vs_pilots_{c}_snr{int(snr)}.csv")
        if os.path.exists(p):
            chans.append((c, read_rows(p)))
    if not chans:
        return None
    tag = f"[snr{int(snr)}]"
    methods = ["LS-linear", "LS-rbf", "MMSE-estimated", "ALMMSE-ideal", "MMSE-ideal", "SR" + tag, "ChannelNet" + tag]
    fig, axs = plt.subplots(1, len(chans), figsize=(6.5 * len(chans), 4.8), squeeze=False)
    for ax, (c, rows) in zip(axs[0], chans):
        P = np.array([r["pilots"] for r in rows])
        order = np.argsort(P)
        for m in methods:
            if m not in rows[0]:
                continue
            y = np.array([r.get(m, np.nan) for r in rows], dtype=float)[order]
            ok = ~np.isnan(y)                      # densities without a trained model are skipped
            if not ok.any():
                continue
            st = _style(m)
            ax.semilogy(P[order][ok], y[ok], color=st.get("color"), marker=st.get("marker") or "o",
                        ls=st.get("ls", "-"), label=st.get("label", m))
        ax.set_xticks(P[order])
        ax.set_xlabel("number of pilots per 72x14 frame")
        ax.set_ylabel("MSE")
        ax.set_title(f"{c}: MSE vs number of pilots at SNR = {int(snr)} dB")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, cfg, f"fig5_mse_vs_pilots_snr{int(snr)}.png")


def plot_example(cfg, channel, pilots):
    p = metrics_path(cfg, f"example_{channel}_p{pilots}.npz")
    if not os.path.exists(p):
        return None
    d = np.load(p)
    imgs = [("ground truth", d["truth"]), ("interpolated LS (input)", d["ls_input"]),
            ("SR output", d["sr"]), ("ChannelNet output", d["channelnet"])]
    fig, axs = plt.subplots(2, 4, figsize=(13, 7), sharey=True)
    vmin, vmax = d["truth"].real.min(), d["truth"].real.max()
    emax = np.abs(d["ls_input"] - d["truth"]).max()
    for j, (t, H) in enumerate(imgs):
        im = axs[0, j].imshow(H.real, aspect="auto", origin="lower", vmin=vmin, vmax=vmax, cmap="viridis")
        axs[0, j].set_title(t)
        if j > 0:
            ie = axs[1, j].imshow(np.abs(H - d["truth"]), aspect="auto", origin="lower", vmin=0, vmax=emax,
                                  cmap="magma")
            axs[1, j].set_title(f"|error|, MSE={np.mean(np.abs(H - d['truth']) ** 2):.2e}")
        else:
            axs[1, j].axis("off")
        axs[1, j].set_xlabel("OFDM symbol")
    axs[0, 0].set_ylabel("sub-carrier")
    axs[1, 1].set_ylabel("sub-carrier")
    fig.colorbar(im, ax=axs[0, :], fraction=0.02)
    fig.colorbar(ie, ax=axs[1, :], fraction=0.02)
    fig.suptitle(f"{channel}, {pilots} pilots, SNR = {float(d['snr']):g} dB (real part), model {d['model']}")
    return _save(fig, cfg, f"example_reconstruction_{channel}_p{pilots}.png")
