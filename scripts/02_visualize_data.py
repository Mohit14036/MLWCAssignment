"""Step 2 - data visualisations (mandatory plot 1 = paper Fig. 2, pilot patterns, LR input example).

    python scripts/02_visualize_data.py --config configs/default.yaml
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import plotting  # noqa: E402
from src.config import base_parser, load_config  # noqa: E402
from src.dataset import get_split, ls_observations  # noqa: E402
from src.estimators import apply_linear, interpolation_matrix  # noqa: E402
from src.ofdm import OFDMGrid, pilot_mask  # noqa: E402
from src.utils import Logger  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    log = Logger(os.path.join(cfg["paths"]["results_dir"], "logs", "02_visualize.log"))
    grid = OFDMGrid.from_config(cfg)
    masks = {p: pilot_mask(cfg, grid, p) for p in cfg["pilots"]["patterns"]}
    log("saved", plotting.plot_pilot_patterns(cfg, masks))
    P = int(cfg["pilots"]["main"])
    snr = cfg["plots"]["example_snr"]
    for ch in cfg["channel"]["models"]:
        H = get_split(cfg, ch, "test", log)
        log("saved", plotting.plot_ground_truth(cfg, H[0], ch))
        hls = ls_observations(cfg, H[:1], ch, "viz", P, snr)
        W = interpolation_matrix(masks[P], cfg["estimators"]["sr_input_interp"])
        Hi = apply_linear(W, hls, grid.shape)[0]
        log("saved", plotting.plot_low_res_example(cfg, H[0], Hi, masks[P], ch, snr, P))


if __name__ == "__main__":
    main()
