#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from adas_planning.io.driving_replay import write_driving_replay_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export driving_replay.v0.1 from perception + planning JSON.")
    parser.add_argument("--perception-json", required=True)
    parser.add_argument("--planning-json", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    document = write_driving_replay_document(
        args.output,
        perception_path=args.perception_json,
        planning_path=args.planning_json,
    )
    print(f"Wrote {args.output} ({len(document.get('frames', []))} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
