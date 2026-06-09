from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_export_module():
    module_path = ROOT / "scripts/export_yolo_onnx.py"
    spec = importlib.util.spec_from_file_location("export_yolo_onnx", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_output_path_uses_imgsz_suffix():
    export_mod = _load_export_module()
    path = export_mod.resolve_output_path(
        Path("outputs/models/adas_yolov8n_bdd100k.pt"),
        Path("outputs/models"),
        640,
        None,
    )
    assert path.name == "adas_yolov8n_bdd100k_640.onnx"


def test_run_planning_demo_default_perception_config():
    demo_mod_path = ROOT / "scripts/run_planning_demo.py"
    spec = importlib.util.spec_from_file_location("run_planning_demo", demo_mod_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DEFAULT_PERCEPTION_CONFIG == "configs/bdd100k_yolo_kind_tuned_post_nms.yaml"
    with patch.object(sys, "argv", ["run_planning_demo.py"]):
        args = module.parse_args()
    assert args.perception_config == module.DEFAULT_PERCEPTION_CONFIG


@pytest.mark.slow
def test_export_yolo_onnx_smoke(tmp_path: Path):
    model_path = ROOT / "outputs/models/adas_yolov8n_bdd100k.pt"
    if not model_path.is_file():
        pytest.skip("finetuned weight not present locally")

    pytest.importorskip("ultralytics")

    output_path = tmp_path / "adas_yolov8n_bdd100k_640.onnx"
    subprocess.run(
        [
            sys.executable,
            "scripts/export_yolo_onnx.py",
            "--model",
            str(model_path),
            "--output-dir",
            str(tmp_path),
            "--output-name",
            output_path.name,
            "--imgsz",
            "640",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
    )
    assert output_path.is_file()
    assert output_path.stat().st_size > 1024
