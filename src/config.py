"""Configuration handling.

* YAML files may contain ``base: other.yaml`` to inherit from another file (deep merge).
* Any entry can be overridden from the command line:  ``--set training.ir.max_epochs=20``.
* Relative paths in ``cfg['paths']`` are resolved against the project root.
"""
from __future__ import annotations

import argparse
import copy
import os

import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG = os.path.join(PROJECT_ROOT, "configs", "default.yaml")


def deep_update(base: dict, upd: dict) -> dict:
    """Recursively merge ``upd`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "base" in cfg:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), cfg.pop("base"))
        cfg = deep_update(_load_yaml(base_path), cfg)
    return cfg


def _apply_override(cfg: dict, expr: str) -> None:
    """Apply one ``a.b.c=value`` override (value parsed as YAML)."""
    if "=" not in expr:
        raise ValueError(f"Override must look like key.sub=value, got: {expr}")
    key, val = expr.split("=", 1)
    node = cfg
    parts = key.strip().split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = yaml.safe_load(val)


def load_config(path: str | None = None, overrides: list[str] | None = None) -> dict:
    cfg = _load_yaml(path or DEFAULT_CONFIG)
    for o in overrides or []:
        _apply_override(cfg, o)
    # pilot-pattern keys must be ints (YAML may already give ints, CLI may give str)
    cfg["pilots"]["patterns"] = {int(k): v for k, v in cfg["pilots"]["patterns"].items()}
    # resolve paths
    for k, p in cfg["paths"].items():
        if not os.path.isabs(p):
            p = os.path.join(PROJECT_ROOT, p)
        cfg["paths"][k] = os.path.normpath(p)
        os.makedirs(cfg["paths"][k], exist_ok=True)
    res = cfg["paths"]["results_dir"]
    for sub in ("figures", "metrics", "logs", "figures/loss_curves"):
        os.makedirs(os.path.join(res, sub), exist_ok=True)
    cfg["_config_path"] = os.path.abspath(path or DEFAULT_CONFIG)
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config file")
    ap.add_argument("--set", dest="overrides", action="append", default=[],
                    help="override a config entry, e.g. --set training.ir.max_epochs=20 (repeatable)")
    return ap


def iter_experiments(cfg: dict, channel=None, pilots=None, snr=None):
    """Yield (channel, n_pilots, train_snr_spec) for every model listed in cfg['experiments'].

    Optional filters restrict the list (used by the CLI of the training script).
    """
    for exp in cfg["experiments"]:
        if channel is not None and exp["channel"] != channel:
            continue
        if pilots is not None and int(exp["pilots"]) != int(pilots):
            continue
        for s in exp["train_snrs"]:
            if snr is not None and str(s) != str(snr):
                continue
            yield exp["channel"], int(exp["pilots"]), s


def experiment_groups(cfg: dict):
    """Unique (channel, pilots) pairs that have at least one trained model."""
    seen = []
    for ch, p, _ in iter_experiments(cfg):
        if (ch, p) not in seen:
            seen.append((ch, p))
    return seen
