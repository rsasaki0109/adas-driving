#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile

from check_bdd100k import check_bdd100k, print_report


OFFICIAL_DOWNLOAD_URL = "https://bdd-data.berkeley.edu/download.html"
OFFICIAL_DOC_URL = "https://doc.bdd100k.com/"
HF_REPO_ID = "Hanshiya/bdd100k"
HF_DATASET_URL = f"https://huggingface.co/datasets/{HF_REPO_ID}"

IMAGE_PATTERNS = (
    "*100k*images*.zip",
    "*bdd100k*images*100k*.zip",
    "*bdd100k*image*100k*.zip",
    "*images*100k*.zip",
    "*image*100k*.zip",
    "*100k*images*.tar*",
    "*bdd100k*images*100k*.tar*",
    "*bdd100k*image*100k*.tar*",
    "*images*100k*.tar*",
    "*image*100k*.tar*",
)

LABEL_PATTERNS = (
    "*bdd100k*det*20*label*.zip",
    "*bdd100k*label*det*20*.zip",
    "*det*20*label*.zip",
    "*label*det*20*.zip",
    "*bdd100k*labels*.zip",
    "*labels*.zip",
    "*bdd100k*det*20*label*.tar*",
    "*bdd100k*label*det*20*.tar*",
    "*det*20*label*.tar*",
    "*label*det*20*.tar*",
    "*bdd100k*labels*.tar*",
    "*labels*.tar*",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare BDD100K-style detection data for adas-perception evaluation."
    )
    parser.add_argument(
        "--data-root",
        default="data/bdd100k",
        help="Target BDD100K root. Expected output: images/100k/val and labels/det_20/det_val.json.",
    )
    parser.add_argument(
        "--download-dir",
        default=None,
        help="Directory containing official downloaded archives. Used to auto-detect archives.",
    )
    parser.add_argument(
        "--download-val",
        action="store_true",
        help="Download and export the public BDD100K validation mirror from Hugging Face.",
    )
    parser.add_argument(
        "--hf-repo",
        default=HF_REPO_ID,
        help="Hugging Face dataset repo used by --download-val.",
    )
    parser.add_argument(
        "--hf-max-images",
        type=int,
        default=None,
        help="Limit Hugging Face validation image downloads. Defaults to all samples.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Parallel image download workers used by --download-val.",
    )
    parser.add_argument("--images-archive", default=None, help="Official 100K Images archive path.")
    parser.add_argument("--labels-archive", default=None, help="Official Detection 2020 Labels archive path.")
    parser.add_argument("--check-only", action="store_true", help="Only validate the prepared data root.")
    parser.add_argument("--force", action="store_true", help="Allow extraction into a non-empty target directory.")
    parser.add_argument("--max-samples", type=int, default=200, help="Number of label image paths to validate.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    images_root = data_root / "images" / "100k" / "val"
    labels_path = data_root / "labels" / "det_20" / "det_val.json"

    if not args.check_only:
        if args.download_val:
            _download_hf_val(
                data_root,
                repo=args.hf_repo,
                max_images=args.hf_max_images,
                workers=max(1, args.workers),
            )
        else:
            archives = _resolve_archives(args)
            if archives:
                data_root.mkdir(parents=True, exist_ok=True)
                _ensure_extractable(data_root, args.force)
                for archive in archives:
                    _extract_archive(archive, data_root)
            else:
                _print_download_instructions(data_root)

    report = check_bdd100k(images_root=images_root, labels_path=labels_path, max_samples=args.max_samples)
    print_report(report)

    if report["ready_for_eval"]:
        print("")
        print("Ready. Example evaluation command:")
        print(
            "python scripts/evaluate_bdd100k.py "
            f"--images-root {images_root} "
            f"--labels {labels_path} "
            "--config configs/bdd100k_eval.yaml "
            "--max-images 500 "
            "--output outputs/bdd100k_val_eval.json"
        )
        return 0

    print("")
    print("Not ready yet. Download the official files, then rerun with --download-dir or explicit archive paths.")
    _print_download_instructions(data_root)
    return 2


def _resolve_archives(args: argparse.Namespace) -> list[Path]:
    archives: list[Path] = []
    if args.images_archive:
        archives.append(Path(args.images_archive))
    if args.labels_archive:
        archives.append(Path(args.labels_archive))
    if archives:
        missing = [str(path) for path in archives if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Archive not found: {', '.join(missing)}")
        return archives

    if not args.download_dir:
        return []

    download_dir = Path(args.download_dir)
    if not download_dir.exists():
        raise FileNotFoundError(f"Download directory not found: {download_dir}")

    image_archive = _find_archive(download_dir, IMAGE_PATTERNS)
    label_archive = _find_archive(download_dir, LABEL_PATTERNS)
    return [path for path in (image_archive, label_archive) if path is not None]


def _find_archive(download_dir: Path, patterns: tuple[str, ...]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(path for path in download_dir.glob(pattern) if path.is_file())
    if not matches:
        return None
    matches = sorted(set(matches), key=lambda path: (len(path.name), path.name))
    return matches[0]


def _download_hf_val(data_root: Path, repo: str, max_images: int | None, workers: int) -> None:
    images_root = data_root / "images" / "100k" / "val"
    labels_path = data_root / "labels" / "det_20" / "det_val.json"
    cache_dir = data_root.parent / "_downloads" / "huggingface" / repo.replace("/", "__")
    samples_path = cache_dir / "samples.json"

    images_root.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading BDD100K validation mirror from https://huggingface.co/datasets/{repo}", flush=True)
    _download_file(_hf_resolve_url(repo, "samples.json"), samples_path)

    with samples_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    samples = list(payload.get("samples", []))
    if max_images is not None:
        samples = samples[:max_images]

    frames = []
    jobs = []
    total = len(samples)
    for sample in samples:
        source_path = str(sample["filepath"])
        image_name = Path(source_path).name
        metadata = sample.get("metadata", {})
        expected_size = int(metadata.get("size_bytes", 0)) or None
        jobs.append((_hf_resolve_url(repo, source_path), images_root / image_name, expected_size))
        frames.append(_sample_to_scalabel_frame(sample, image_name))

    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_download_file, url, destination, expected_size, False)
            for url, destination, expected_size in jobs
        ]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed == total or completed % 100 == 0:
                print(f"Downloaded/exported {completed}/{total} images", flush=True)

    with labels_path.open("w", encoding="utf-8") as f:
        json.dump(frames, f, ensure_ascii=False)
        f.write("\n")
    print(f"Saved {labels_path}", flush=True)


def _hf_resolve_url(repo: str, path: str) -> str:
    quoted = urllib.parse.quote(path)
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{quoted}"


def _sample_to_scalabel_frame(sample: dict, image_name: str) -> dict:
    metadata = sample.get("metadata", {})
    width = float(metadata.get("width", 0) or 0)
    height = float(metadata.get("height", 0) or 0)
    labels = []
    for index, detection in enumerate(sample.get("detections", {}).get("detections", [])):
        box = _normalized_box_to_box2d(detection.get("bounding_box", []), width, height)
        if box is None:
            continue
        labels.append(
            {
                "id": str(detection.get("_id", {}).get("$oid", index)),
                "category": str(detection.get("label", "")),
                "attributes": {
                    "occluded": bool(detection.get("occluded", False)),
                    "truncated": bool(detection.get("truncated", False)),
                    "trafficLightColor": _normalize_light_color(detection.get("trafficLightColor")),
                },
                "box2d": box,
            }
        )
    return {
        "name": image_name,
        "attributes": {
            "weather": _classification_label(sample.get("weather")),
            "timeofday": _classification_label(sample.get("timeofday")),
            "scene": _classification_label(sample.get("scene")),
        },
        "labels": labels,
    }


def _normalized_box_to_box2d(box: list, width: float, height: float) -> dict | None:
    if len(box) != 4 or width <= 0 or height <= 0:
        return None
    x, y, w, h = [float(value) for value in box]
    return {
        "x1": x * width,
        "y1": y * height,
        "x2": (x + w) * width,
        "y2": (y + h) * height,
    }


def _normalize_light_color(value: object) -> str:
    mapping = {
        "r": "red",
        "red": "red",
        "y": "yellow",
        "yellow": "yellow",
        "g": "green",
        "green": "green",
    }
    return mapping.get(str(value).strip().lower(), "unknown")


def _classification_label(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("label", "unknown"))
    return "unknown"


def _download_file(
    url: str,
    destination: Path,
    expected_size: int | None = None,
    verbose: bool = True,
) -> None:
    if destination.exists() and _existing_download_ok(destination, expected_size):
        if verbose:
            print(f"Using existing {destination}", flush=True)
        return
    if verbose:
        print(f"Downloading {url}", flush=True)
    last_error: Exception | None = None
    temp_path = destination.with_name(f"{destination.name}.part")
    for attempt in range(1, 4):
        try:
            if temp_path.exists():
                temp_path.unlink()
            if shutil.which("curl"):
                _download_with_curl(url, temp_path)
            else:
                _download_with_urllib(url, temp_path)
            temp_path.replace(destination)
            if not _existing_download_ok(destination, expected_size):
                raise RuntimeError(f"Incomplete download for {destination}")
            return
        except Exception as exc:
            last_error = exc
            if destination.exists():
                destination.unlink()
            if temp_path.exists():
                temp_path.unlink()
            if attempt < 3:
                time.sleep(1.5 * attempt)
    raise RuntimeError(f"Failed to download {url}: {last_error}") from last_error


def _download_with_curl(url: str, destination: Path) -> None:
    subprocess.run(
        [
            "curl",
            "-L",
            "--fail",
            "--silent",
            "--show-error",
            "--connect-timeout",
            "20",
            "--max-time",
            "120",
            "-o",
            str(destination),
            url,
        ],
        check=True,
        timeout=135,
    )


def _download_with_urllib(url: str, destination: Path) -> None:
    with urllib.request.urlopen(url, timeout=45) as response, destination.open("wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def _existing_download_ok(path: Path, expected_size: int | None) -> bool:
    size = path.stat().st_size
    if expected_size is not None:
        return size == expected_size
    return size > 0


def _ensure_extractable(data_root: Path, force: bool) -> None:
    expected_children = {"images", "labels"}
    existing = {path.name for path in data_root.iterdir()}
    unexpected = sorted(existing - expected_children)
    if unexpected and not force:
        names = ", ".join(unexpected[:8])
        raise RuntimeError(
            f"{data_root} is not empty ({names}). Pass --force if this is the intended BDD100K root."
        )


def _extract_archive(archive: Path, data_root: Path) -> None:
    print(f"Extracting {archive}")
    target = data_root.parent if _archive_has_bdd100k_root(archive) else data_root
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                _check_member_path(target, member.filename)
            zf.extractall(target)
        return

    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            for member in tf.getmembers():
                _check_member_path(target, member.name)
            tf.extractall(target)
        return

    raise ValueError(f"Unsupported archive format: {archive}")


def _archive_has_bdd100k_root(archive: Path) -> bool:
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            return any(name.split("/", 1)[0] == "bdd100k" for name in zf.namelist())
    if tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            return any(member.name.split("/", 1)[0] == "bdd100k" for member in tf.getmembers())
    return False


def _check_member_path(target: Path, member_name: str) -> None:
    destination = (target / member_name).resolve()
    root = target.resolve()
    if root not in destination.parents and destination != root:
        raise RuntimeError(f"Unsafe archive member path: {member_name}")


def _print_download_instructions(data_root: Path) -> None:
    print("")
    print("BDD100K official download required:")
    print(f"- Download page: {OFFICIAL_DOWNLOAD_URL}")
    print(f"- Documentation: {OFFICIAL_DOC_URL}")
    print(f"- Public validation mirror used by --download-val: {HF_DATASET_URL}")
    print("- Required for this repo's object detection evaluation:")
    print("  - 100K Images")
    print("  - Detection 2020 Labels")
    print(f"- Target root: {data_root}")
    print("- Expected paths after extraction:")
    print(f"  - {data_root / 'images' / '100k' / 'val'}")
    print(f"  - {data_root / 'labels' / 'det_20' / 'det_val.json'}")
    print("- Example after manual download:")
    print(f"  python scripts/prepare_bdd100k.py --download-val --data-root {data_root}")
    print(f"  python scripts/prepare_bdd100k.py --download-dir ~/Downloads --data-root {data_root}")


if __name__ == "__main__":
    raise SystemExit(main())
