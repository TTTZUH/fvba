"""Configuration loading and validation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


_REQUIRED_SECTIONS = ("experiment", "runtime", "data", "attack")


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {context}.{key}")
    return mapping[key]


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the cross-cutting invariants used by every experiment."""
    for section in _REQUIRED_SECTIONS:
        if section not in config or not isinstance(config[section], Mapping):
            raise ValueError(f"Missing required configuration section: {section}")

    experiment = config["experiment"]
    runtime = config["runtime"]
    data = config["data"]
    attack = config["attack"]
    for key in ("name", "method", "full_run"):
        _require(experiment, key, "experiment")
    if experiment["method"] not in {"fpba", "vtba"}:
        raise ValueError("experiment.method must be 'fpba' or 'vtba'")
    for key in ("seed", "device", "allow_cpu_full_run"):
        _require(runtime, key, "runtime")
    num_classes = int(_require(data, "num_classes", "data"))
    if num_classes < 2:
        raise ValueError("data.num_classes must be at least 2")

    epsilon = float(_require(attack, "epsilon", "attack"))
    if not 0.0 <= epsilon <= 1.0:
        raise ValueError("attack.epsilon must lie in [0, 1]")
    if int(_require(attack, "carriers_per_class", "attack")) <= 0:
        raise ValueError("attack.carriers_per_class must be positive")
    dct = _require(attack, "dct", "attack")
    if not isinstance(dct, Mapping):
        raise ValueError("attack.dct must be a mapping")
    block_size = int(_require(dct, "block_size", "attack.dct"))
    low_freq_size = int(_require(dct, "low_freq_size", "attack.dct"))
    if block_size <= 0:
        raise ValueError("attack.dct.block_size must be positive")
    if not 0 <= low_freq_size <= block_size:
        raise ValueError("attack.dct.low_freq_size must lie in [0, block_size]")
    if float(_require(dct, "strength", "attack.dct")) < 0:
        raise ValueError("attack.dct.strength must be non-negative")


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML file and validate its shared schema."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(loaded)
    return loaded

