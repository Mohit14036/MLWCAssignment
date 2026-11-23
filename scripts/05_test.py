"""Step 5 - TESTING (Task 6). Loads the saved checkpoints; never trains.

* evaluates every trained ChannelNet (and its SR-only output) over the test-SNR list
* merges with the conventional-estimator results of step 3 (runs step 3 for missing tables)
* builds the MSE-vs-#pilots tables at the fixed pilot-sweep SNR

    python scripts/05_test.py --config configs/default.yaml
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import base_parser, experiment_groups, load_config  # noqa: E402
from src.evaluation import (evaluate_channelnet, evaluate_conventional, merge_tables,  # noqa: E402
                            metrics_path, pilot_sweep_table)
from src.utils import Logger, get_device, set_global_seed  # noqa: E402


def main():
    args = base_parser(__doc__).parse_args()
    cfg = load_config(args.config, args.overrides)
    log = Logger(os.path.join(cfg["paths"]["results_dir"], "logs", "05_test.log"))
    set_global_seed(int(cfg["seed"]), bool(cfg["training"].get("deterministic", True)))
    device = get_device(cfg)
    snrs = cfg["snr"]["test"]
    reuse = bool(cfg.get("test", {}).get("skip_dl_if_present", False))
    for ch, p in experiment_groups(cfg):
        if reuse and os.path.exists(metrics_path(cfg, f"channelnet_{ch}_p{p}.csv")):
            log(f"=== ChannelNet test: {ch}, {p} pilots -> reusing existing channelnet_{ch}_p{p}.csv ===")
            continue
        log(f"=== ChannelNet test: {ch}, {p} pilots ===")
        evaluate_channelnet(cfg, ch, p, snrs, device, log)
    # merge with conventional baselines for every (channel, pilots) combination
    for ch in cfg["channel"]["models"]:
        for p in cfg["pilots"]["patterns"]:
            if not os.path.exists(metrics_path(cfg, f"conventional_{ch}_p{p}.csv")):
                log(f"conventional table missing for {ch} P={p} -> computing")
                evaluate_conventional(cfg, ch, p, snrs, log)
            merge_tables(cfg, ch, p)
        rows = pilot_sweep_table(cfg, ch, cfg["snr"]["pilot_sweep"])
        log(f"{ch}: MSE vs pilots at {cfg['snr']['pilot_sweep']} dB:")
        for r in rows:
            log("   ", {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in r.items()})
    log("done.")


if __name__ == "__main__":
    main()
