from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into base and return a new dict."""
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}
    if config_path.resolve() == DEFAULT_CONFIG_PATH.resolve():
        return loaded
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        base = yaml.safe_load(f) or {}
    return deep_update(base, loaded)


def apply_runtime_overrides(
    config: dict[str, Any],
    *,
    device: str | None = None,
    disable_objects: bool = False,
) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if device:
        overrides.setdefault("objects", {})["device"] = device
    if disable_objects:
        overrides.setdefault("objects", {})["enabled"] = False
    return deep_update(config, overrides)
