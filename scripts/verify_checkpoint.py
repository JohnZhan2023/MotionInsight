#!/usr/bin/env python3
"""Validate that a local MotionInsight checkpoint contains release-time files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_FILES = {
    "added_tokens.json",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
}
REQUIRED_MOTION_KEYS = {
    "model.motion_projector.0.weight",
    "model.camera_motion_projector.0.weight",
    "model.motion_fusion.0.weight",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()

    missing = sorted(name for name in REQUIRED_FILES if not (checkpoint / name).is_file())
    if missing:
        raise SystemExit("Missing checkpoint files: " + ", ".join(missing))

    with (checkpoint / "config.json").open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("architectures") != ["Qwen3VLForConditionalGeneration"]:
        raise SystemExit("config.json does not define the expected Qwen3-VL architecture")
    if config.get("motion_token_id") is None:
        raise SystemExit("config.json does not define motion_token_id")

    with (checkpoint / "model.safetensors.index.json").open(encoding="utf-8") as handle:
        index = json.load(handle)
    weight_map = index.get("weight_map", {})
    missing_keys = sorted(REQUIRED_MOTION_KEYS - weight_map.keys())
    if missing_keys:
        raise SystemExit("Missing motion weights: " + ", ".join(missing_keys))
    missing_shards = sorted({name for name in weight_map.values() if not (checkpoint / name).is_file()})
    if missing_shards:
        raise SystemExit("Missing model shards: " + ", ".join(missing_shards))

    print(f"Checkpoint is complete: {checkpoint}")
    print(f"Weight tensors: {len(weight_map)}")
    print(f"Model shards: {len(set(weight_map.values()))}")


if __name__ == "__main__":
    main()
