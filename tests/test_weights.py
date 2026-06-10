from __future__ import annotations

from pathlib import Path

from adas_perception import weights


def test_existing_file_is_kept(tmp_path: Path):
    target = tmp_path / "adas_yolov8n_bdd100k.pt"
    target.write_bytes(b"local model")
    assert weights.ensure_weight(target) is True
    assert target.read_bytes() == b"local model"  # never overwritten


def test_unknown_weight_is_not_downloaded(tmp_path: Path):
    assert weights.ensure_weight(tmp_path / "my_custom_model.pt") is False


def test_sha256_mismatch_discards_download(tmp_path: Path, monkeypatch):
    target = tmp_path / "traffic_light_state.onnx"

    def fake_urlretrieve(url: str, dest):
        Path(dest).write_bytes(b"corrupted payload")

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", fake_urlretrieve)
    assert weights.ensure_weight(target) is False
    assert not target.exists()
    assert not target.with_suffix(".onnx.part").exists()


def test_download_failure_returns_false(tmp_path: Path, monkeypatch):
    def fake_urlretrieve(url: str, dest):
        raise OSError("network down")

    monkeypatch.setattr(weights.urllib.request, "urlretrieve", fake_urlretrieve)
    assert weights.ensure_weight(tmp_path / "traffic_light_state.onnx") is False


def test_ensure_config_weights_walks_nested_config(tmp_path: Path, monkeypatch):
    requested: list[str] = []
    monkeypatch.setattr(
        weights, "ensure_weight", lambda path: requested.append(str(path)) or True
    )
    weights.ensure_config_weights(
        {
            "objects": {"model": "outputs/models/adas_yolov8n_bdd100k.pt"},
            "lane": {"segmentation": {"model_path": "outputs/models/twinlitenet_lane.onnx"}},
            "list": [{"model": "outputs/models/traffic_light_state.onnx"}],
            "other": {"model": "outputs/models/unrelated_model.onnx"},
        }
    )
    assert len(requested) == 3  # unrelated_model is not a known weight
