"""Small shared helpers: seeding, device, logging, naming conventions."""
from __future__ import annotations

import json
import os
import random
import sys
import time
import zlib

import numpy as np


# ----------------------------------------------------------------------------- seeding
def stable_int(x) -> int:
    """Deterministic (process-independent) 32-bit integer for any key (str / int / float)."""
    return zlib.crc32(str(x).encode("utf-8")) & 0xFFFFFFFF


def make_rng(cfg: dict, *keys) -> np.random.Generator:
    """Independent NumPy RNG stream derived from the master seed and a tuple of keys.

    e.g. make_rng(cfg, 'noise', 'VehA', 'test', 48, 22) always produces the same noise,
    so conventional and deep-learning estimators are evaluated on identical observations.
    """
    entropy = [int(cfg["seed"])] + [stable_int(k) for k in keys]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed python / numpy / torch (CPU + CUDA) and optionally force deterministic kernels."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:  # pragma: no cover
            pass


def get_device(cfg: dict):
    import torch
    want = cfg["training"].get("device", "auto")
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    n = int(cfg["training"].get("num_threads", 0) or 0)
    if n > 0:
        torch.set_num_threads(n)
    return torch.device(want)


# ----------------------------------------------------------------------------- naming
def snr_tag(snr) -> str:
    """'snr12', 'snr22', 'snrmix' - used in file names and table columns."""
    if isinstance(snr, str):
        return f"snr{snr}"
    return f"snr{int(snr) if float(snr).is_integer() else snr}"


def ckpt_dir(cfg: dict, channel: str, pilots: int, snr) -> str:
    d = os.path.join(cfg["paths"]["ckpt_dir"], channel, f"p{pilots}", snr_tag(snr))
    os.makedirs(d, exist_ok=True)
    return d


def history_path(cfg: dict, channel: str, pilots: int, snr, stage: str) -> str:
    return os.path.join(cfg["paths"]["results_dir"], "logs",
                        f"{channel}_p{pilots}_{snr_tag(snr)}_{stage}_history.csv")


# ----------------------------------------------------------------------------- io / logging
def json_safe(obj):
    """Round-trip through JSON so only plain python types remain (safe for torch.load(weights_only=True))."""
    return json.loads(json.dumps(obj, default=str))


def save_json(obj, path: str) -> None:
    def conv(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=conv)


class Logger:
    """Print to stdout and append to a log file at the same time."""

    def __init__(self, path: str | None = None):
        self.path = path
        if path:
            os.makedirs(os.path.dirname(path), exist_ok=True)

    def __call__(self, *msg):
        line = " ".join(str(m) for m in msg)
        stamp = time.strftime("%H:%M:%S")
        print(f"[{stamp}] {line}", flush=True)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {line}\n")


def env_summary() -> dict:
    import platform
    import scipy
    import torch
    d = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": str(np.__version__),
        "scipy": str(scipy.__version__),
        "torch": str(torch.__version__),
        "cuda": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        d["gpu"] = torch.cuda.get_device_name(0)
    return d
