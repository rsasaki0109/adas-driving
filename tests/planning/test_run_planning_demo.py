from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("compare_configs", [False, True])
def test_run_planning_demo_exports_driving_replay(tmp_path: Path, compare_configs: bool):
    output_dir = tmp_path / "demo"
    cmd = [
        sys.executable,
        "scripts/run_planning_demo.py",
        "--output-dir",
        str(output_dir),
        "--max-frames",
        "3",
    ]
    if compare_configs:
        cmd.extend(["--compare-configs", "--export-benchmark"])
    subprocess.run(cmd, cwd=ROOT, check=True)

    summary_path = output_dir / "run_summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert Path(summary["planning_json"]).exists()
    assert Path(summary["overlay_mp4"]).exists()
    assert Path(summary["driving_replay_json"]).exists()

    replay_payload = json.loads(Path(summary["driving_replay_json"]).read_text(encoding="utf-8"))
    assert replay_payload["schema_version"] == "driving_replay.v0.1"
    assert replay_payload["frames"]
    assert "planning" in replay_payload["frames"][0]

    if compare_configs:
        assert Path(summary["config_compare_json"]).exists()
        assert Path(summary["benchmark_csv"]).exists()
        assert Path(summary["benchmark_markdown"]).exists()
