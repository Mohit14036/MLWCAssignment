"""Step 6 - all result figures (reads results/metrics + results/logs; no computation).

    python scripts/06_make_plots.py --config configs/default.yaml

Figures (results/figures):
    loss_curves/loss_<ch>_p<P>_snr<S>.png    training/validation loss, SR + IR stages   (mandatory 2)
    fig3_VehA_mse_vs_snr.png                  paper Fig. 3: MSE vs SNR, VehA, 48 pilots  (mandatory 3)
    fig4_SUI5_mse_vs_snr.png                  paper Fig. 4: MSE vs SNR, SUI5, 48 pilots  (mandatory 3)
    fig5_mse_vs_pilots_snr20.png              paper Fig. 5: MSE vs #pilots at 20 dB      (mandatory 3 + 5)
    high_snr_VehA_vs_SUI5.png                 separate VehA / SUI5, 22 dB-trained model  (mandatory 4)
    snr_generalization.png                    gap to ideal MMSE of 12 / 22 / mix models
    example_reconstruction_<ch>_p<P>.png      images: truth / input / SR / ChannelNet
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import plotting  # noqa: E402
from src.config import base_parser, iter_experiments, load_config  # noqa: E402
from src.evaluation import read_rows  # noqa: E402
from src.utils import Logger, history_path, snr_tag  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    log = Logger(os.path.join(cfg["paths"]["results_dir"], "logs", "06_plots.log"))
    out = []
    # loss curves
    for ch, p, s in iter_experiments(cfg):
        hs, hi = history_path(cfg, ch, p, s, "sr"), history_path(cfg, ch, p, s, "ir")
        if os.path.exists(hs) or os.path.exists(hi):
            out.append(plotting.plot_loss_curves(cfg, ch, p, snr_tag(s),
                                                 read_rows(hs) if os.path.exists(hs) else [],
                                                 read_rows(hi) if os.path.exists(hi) else []))
    # paper-style figures 3, 4, 5
    for name in ("fig3", "fig4"):
        spec = cfg["plots"][name]
        out.append(plotting.plot_mse_vs_snr_figure(cfg, name, spec["channel"], spec["pilots"]))
    for extra in cfg["plots"].get("extra", []):          # optional additional MSE-vs-SNR figures
        out.append(plotting.plot_mse_vs_snr_figure(cfg, extra["name"], extra["channel"], extra["pilots"]))
    P = int(cfg["pilots"]["main"])
    out.append(plotting.plot_high_snr_comparison(cfg, P))
    out.append(plotting.plot_generalization(cfg, P))
    out.append(plotting.plot_pilot_sweep(cfg, cfg["snr"]["pilot_sweep"]))
    for ch in cfg["channel"]["models"]:
        out.append(plotting.plot_example(cfg, ch, P))
    for o in out:
        if o:
            log("saved", o)


if __name__ == "__main__":
    main()
