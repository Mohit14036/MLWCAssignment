"""Run the complete ChannelNet pipeline end to end.

    python run_all.py --config configs/final.yaml      # submitted set-up (GPU recommended)
    python run_all.py --config configs/final.yaml --steps test plots   # evaluate saved checkpoints only
    python run_all.py --set training.ir.max_epochs=30  # overrides are forwarded to every step

Steps: data -> viz -> conventional -> train -> test -> plots -> report
Every step is idempotent (cached data / existing checkpoints are reused), so an interrupted run
can simply be restarted.
"""
import argparse
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
STEPS = {
    "sanity": "tests/test_sanity.py",
    "data": "scripts/01_generate_data.py",
    "viz": "scripts/02_visualize_data.py",
    "conventional": "scripts/03_conventional.py",
    "train": "scripts/04_train.py",
    "test": "scripts/05_test.py",
    "plots": "scripts/06_make_plots.py",
    "report": "scripts/07_build_report.py",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "default.yaml"))
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    ap.add_argument("--set", dest="overrides", action="append", default=[])
    args = ap.parse_args()
    extra = []
    for o in args.overrides:
        extra += ["--set", o]
    t_all = time.time()
    for step in args.steps:
        if not os.path.exists(os.path.join(ROOT, STEPS[step])):      # optional step not shipped
            print(f">>> step {step}: {STEPS[step]} not present - skipped", flush=True)
            continue
        cmd = [sys.executable, os.path.join(ROOT, STEPS[step])]
        if step != "sanity":
            cmd += ["--config", args.config] + extra
        print(f"\n{'=' * 80}\n>>> STEP {step}: {' '.join(cmd)}\n{'=' * 80}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            sys.exit(f"step '{step}' failed with exit code {r.returncode}")
        print(f">>> step {step} finished in {(time.time() - t0) / 60:.1f} min", flush=True)
    print(f"\nALL DONE in {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
