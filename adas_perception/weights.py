"""Known model weights: download-on-demand so demos run right after clone.

`outputs/models/*` is gitignored; the project weights are published as
GitHub Release assets instead, and third-party weights are fetched from
their upstream repos. Every entry is sha256-pinned. Downloads only trigger
for paths whose basename is a known weight and whose file is missing, so
custom/local models are never touched.
"""
from __future__ import annotations

import hashlib
import urllib.request
from pathlib import Path
from typing import Any

RELEASE_BASE = "https://github.com/rsasaki0109/adas-driving/releases/download/v0.1.0"

KNOWN_WEIGHTS: dict[str, dict[str, str]] = {
    "adas_yolov8n_bdd100k.pt": {
        "url": f"{RELEASE_BASE}/adas_yolov8n_bdd100k.pt",
        "sha256": "dc5f6ff1d7410f629e355d32d513aa629624343d8d64f9a057a83a5c5c2e669f",
    },
    "traffic_light_state.onnx": {
        "url": f"{RELEASE_BASE}/traffic_light_state.onnx",
        "sha256": "50c0e2d902bcb753a3d7303ec53729acb0d991de04f449e67d41caa148a30044",
    },
    # TwinLiteNet (MIT) is fetched from its upstream repo.
    "twinlitenet_lane.onnx": {
        "url": "https://raw.githubusercontent.com/harrylal/TwinLiteNet-onnxruntime/main/models/best.onnx",
        "sha256": "f505b56bd7c9e4e38b2928373aa4daa191a9a398e07e1fb340024b9ce84d4285",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_weight(path: str | Path) -> bool:
    """Download `path` if missing and its basename is a known weight.

    Returns True when the file exists afterwards (already present or
    downloaded), False when the weight is unknown or the download failed.
    Never raises: callers keep their own missing-model fallbacks.
    """
    target = Path(path)
    if target.is_file():
        return True
    entry = KNOWN_WEIGHTS.get(target.name)
    if entry is None:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_suffix(target.suffix + ".part")
    try:
        print(f"[weights] downloading {target.name} from {entry['url']}")
        urllib.request.urlretrieve(entry["url"], temp)  # noqa: S310 - pinned https URLs
        actual = _sha256(temp)
        if actual != entry["sha256"]:
            temp.unlink(missing_ok=True)
            print(f"[weights] sha256 mismatch for {target.name} (got {actual}); discarded")
            return False
        temp.replace(target)
        print(f"[weights] saved {target}")
        return True
    except Exception as error:  # noqa: BLE001 - downloads are best-effort
        temp.unlink(missing_ok=True)
        print(f"[weights] download failed for {target.name}: {error}")
        return False


def ensure_config_weights(config: Any) -> None:
    """Walk a loaded YAML config and ensure every referenced known weight."""
    if isinstance(config, dict):
        for value in config.values():
            ensure_config_weights(value)
    elif isinstance(config, list):
        for value in config:
            ensure_config_weights(value)
    elif isinstance(config, str) and Path(config).name in KNOWN_WEIGHTS:
        ensure_weight(config)
