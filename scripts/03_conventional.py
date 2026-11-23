"""Step 3 - conventional channel estimation (Task 2): LS + 2-D interpolation and MMSE variants.

Evaluates every channel model x every pilot pattern over the test SNR list.

    python scripts/03_conventional.py --config configs/default.yaml
    python scripts/03_conventional.py --channel VehA --pilots 48

Outputs results/metrics/conventional_<channel>_p<P>.csv
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import base_parser, load_config  # noqa: E402
from src.evaluation import evaluate_conventional  # noqa: E402
from src.utils import Logger  # noqa: E402


def main():
    ap = base_parser(__doc__)
    ap.add_argument("--channel", default=None)
    ap.add_argument("--pilots", type=int, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config, args.overrides)
    log = Logger(os.path.join(cfg["paths"]["results_dir"], "logs", "03_conventional.log"))
    chans = [args.channel] if args.channel else cfg["channel"]["models"]
    pils = [args.pilots] if args.pilots else sorted(cfg["pilots"]["patterns"], reverse=True)
    for ch in chans:
        for p in pils:
            log(f"=== conventional estimators: {ch}, {p} pilots ===")
            evaluate_conventional(cfg, ch, p, cfg["snr"]["test"], log)


if __name__ == "__main__":
    main()
