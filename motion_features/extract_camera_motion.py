#!/usr/bin/env python3
"""Extract MotionInsight camera poses with VIPE."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import cv2
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
VIPE_ROOT = ROOT / "thirdparty" / "vipe"
sys.path.insert(0, str(VIPE_ROOT))

from vipe.pipeline import make_pipeline
from vipe.streams.base import StreamList


class CameraMotionExtractor:
    def __init__(self, device: str = "cuda", pipeline: str = "default") -> None:
        self.device = device
        self.pipeline = pipeline
        if device.startswith("cuda"):
            torch.cuda.set_device(0 if device == "cuda" else int(device.split(":", 1)[1]))

    def pipeline_config(self):
        load = OmegaConf.load
        config = load(VIPE_ROOT / "configs" / "pipeline" / f"{self.pipeline}.yaml")
        merged = OmegaConf.create()
        for item in list(config.get("defaults", [])):
            if item == "default":
                base = load(VIPE_ROOT / "configs" / "pipeline" / "default.yaml")
                base.pop("defaults", None)
                merged = OmegaConf.merge(merged, base)
            elif isinstance(item, Mapping) and item.get("/slam") is not None:
                slam = load(VIPE_ROOT / "configs" / "slam" / f"{item['/slam']}.yaml")
                merged = OmegaConf.merge(merged, {"slam": slam})
        config.pop("defaults", None)
        merged = OmegaConf.merge(merged, config)
        merged.slam.visualize = False
        merged.output.save_artifacts = False
        merged.output.save_viz = False
        merged.output.save_slam_map = False
        merged.output.skip_exists = False
        merged.post.depth_align_model = None
        OmegaConf.resolve(merged)
        return merged

    def extract(self, video_path: Path) -> torch.Tensor:
        capture = cv2.VideoCapture(str(video_path))
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        capture.release()
        if total_frames <= 0:
            raise RuntimeError(f"Could not read video: {video_path}")

        stream_config = OmegaConf.load(VIPE_ROOT / "configs" / "streams" / "raw_mp4_stream.yaml")
        stream_config.base_path = str(video_path.resolve())
        stream = StreamList.make(stream_config)[0]
        config = self.pipeline_config()
        config.output.path = str(video_path.parent / ".vipe_unused")
        pipeline = make_pipeline(config)
        pipeline.return_payload = True
        output = pipeline.run(stream)
        pose = output.payload.get_view_trajectory(0).matrix().cpu().float()
        if pose.shape[0] != total_frames:
            raise RuntimeError(f"VIPE returned {pose.shape[0]} poses for a {total_frames}-frame video")
        return pose


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pipeline", default="default")
    args = parser.parse_args()
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    pose = CameraMotionExtractor(args.device, args.pipeline).extract(args.video)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(pose, args.output)
    print(f"Saved camera motion to {args.output}: pose={tuple(pose.shape)}")


if __name__ == "__main__":
    main()

