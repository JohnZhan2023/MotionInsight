#!/usr/bin/env python3
"""Install the MotionInsight Qwen3-VL implementation into Transformers 4.57.1."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import shutil
from pathlib import Path


SUPPORTED_VERSION = "4.57.1"
TARGET_PATH = Path("transformers/models/qwen3_vl/modeling_qwen3_vl.py")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def paths() -> tuple[Path, Path, Path]:
    try:
        distribution = importlib.metadata.distribution("transformers")
    except importlib.metadata.PackageNotFoundError as exc:
        raise SystemExit("transformers is not installed. Run: pip install -r requirements.txt") from exc
    if distribution.version != SUPPORTED_VERSION:
        raise SystemExit(f"Expected transformers=={SUPPORTED_VERSION}, found {distribution.version}")
    source = Path(__file__).resolve().parents[1] / "patches" / "modeling_qwen3_vl.py"
    target = Path(distribution.locate_file(TARGET_PATH)).resolve()
    backup = target.with_suffix(target.suffix + ".motioninsight.bak")
    return source, target, backup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    source, target, backup = paths()

    if args.check:
        installed = source.is_file() and target.is_file() and digest(source) == digest(target)
        print(f"MotionInsight patch installed: {installed}")
        print(f"Target: {target}")
        raise SystemExit(0 if installed else 1)
    if args.restore:
        if not backup.is_file():
            raise SystemExit(f"No backup found: {backup}")
        shutil.copy2(backup, target)
        print(f"Restored: {target}")
        return
    if not source.is_file() or not target.is_file():
        raise SystemExit(f"Missing source or target: {source}, {target}")
    if digest(source) == digest(target):
        print(f"MotionInsight patch is already installed: {target}")
        return
    if not backup.exists():
        shutil.copy2(target, backup)
        print(f"Backup: {backup}")
    shutil.copy2(source, target)
    print(f"Installed MotionInsight implementation: {target}")
    print("Restart existing Python processes before inference.")


if __name__ == "__main__":
    main()

