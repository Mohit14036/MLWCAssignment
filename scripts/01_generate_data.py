"""Step 1 - generate (or load from Vienna .mat) the ground-truth channel data sets.

    python scripts/01_generate_data.py --config configs/default.yaml

Outputs  data/channels_<model>_<source>_N<N>_seed<seed>.npy   (N, 72, 14) complex64
         results/metrics/channel_statistics.json               PDP / Doppler / power checks
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from src.channel_models import channel_summary  # noqa: E402
from src.config import base_parser, load_config  # noqa: E402
from src.dataset import load_channels, split_channels  # noqa: E402
from src.ofdm import OFDMGrid  # noqa: E402
from src.utils import Logger, env_summary, save_json  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    log = Logger(os.path.join(cfg["paths"]["results_dir"], "logs", "01_generate_data.log"))
    grid = OFDMGrid.from_config(cfg)
    log("environment:", env_summary())
    log(f"grid {grid.shape}, fs = {grid.fs / 1e6:.2f} MHz, frame = {grid.frame_length} samples "
        f"({grid.frame_length / grid.fs * 1e3:.3f} ms)")
    stats = {}
    for ch in cfg["channel"]["models"]:
        H = load_channels(cfg, ch, log)
        sp = split_channels(cfg, H)
        s = channel_summary(ch, cfg["channel"], grid)      # nominal PDP / Doppler parameters
        s["source"] = cfg["channel"]["source"]
        s.update({
            "n_total": int(H.shape[0]),
            "n_train": int(sp["train"].shape[0]), "n_val": int(sp["val"].shape[0]),
            "n_test": int(sp["test"].shape[0]),
            "mean_power_E|H|^2": float(np.mean(np.abs(H) ** 2)),
            "real_std": float(H.real.std()), "imag_std": float(H.imag.std()),
        })
        stats[ch] = s
        log(ch, s)
    save_json(stats, os.path.join(cfg["paths"]["results_dir"], "metrics", "channel_statistics.json"))
    log("done.")


if __name__ == "__main__":
    main()
