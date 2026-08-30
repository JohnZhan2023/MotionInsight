#!/usr/bin/env python3
"""Extract MotionInsight object-motion features with SAM3 and CoTracker3."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import imageio.v3 as iio
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
for dependency in (ROOT / "thirdparty" / "co-tracker", ROOT / "thirdparty" / "sam3"):
    sys.path.insert(0, str(dependency))

from cotracker.predictor import CoTrackerPredictor
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


class ObjectMotionExtractor:
    def __init__(
        self,
        sam_checkpoint: Path,
        cotracker_checkpoint: Path,
        device: str = "cuda",
        mask_score_threshold: float = 0.8,
    ) -> None:
        self.device = device
        self.mask_score_threshold = mask_score_threshold
        sam_model = build_sam3_image_model(checkpoint_path=str(sam_checkpoint), device=device)
        self.sam_processor = Sam3Processor(sam_model)
        self.cotracker = CoTrackerPredictor(checkpoint=str(cotracker_checkpoint), offline=True).to(device)

    def extract(self, video_path: Path, target: str) -> dict[str, torch.Tensor]:
        frames = iio.imread(video_path, plugin="FFMPEG")
        video = torch.as_tensor(frames).permute(0, 3, 1, 2)[None].float().to(self.device)
        total_frames = int(video.shape[1])
        candidate_frames = [0, total_frames // 2, total_frames - 1, total_frames // 4, 3 * total_frames // 4]

        mask = None
        query_frame = None
        for frame_id in dict.fromkeys(candidate_frames):
            image_tensor = video[0, frame_id]
            image = Image.fromarray(image_tensor.permute(1, 2, 0).cpu().byte().numpy())
            state = self.sam_processor.set_image(image)
            output = self.sam_processor.set_text_prompt(state=state, prompt=target)
            masks, scores = output["masks"], output["scores"]
            if masks is not None and len(masks) > 0 and float(scores.max()) > self.mask_score_threshold:
                best = int(torch.argmax(scores))
                mask = masks[best].unsqueeze(0).float()
                query_frame = frame_id
                break
        if mask is None or query_frame is None:
            raise RuntimeError(f"SAM3 could not find target {target!r} in {video_path}")

        for grid_size in (10, 30, 90):
            _, _, confidence, features = self.cotracker(
                video,
                segm_mask=mask,
                grid_size=grid_size,
                grid_query_frame=query_frame,
            )
            if features.shape[1] > 0:
                return {"x": features.cpu(), "confidence": confidence.cpu()}
        raise RuntimeError(f"CoTracker found no valid tracks for target {target!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--sam-checkpoint", type=Path, required=True)
    parser.add_argument("--cotracker-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-score-threshold", type=float, default=0.8)
    args = parser.parse_args()

    for path in (args.video, args.sam_checkpoint, args.cotracker_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    extractor = ObjectMotionExtractor(
        args.sam_checkpoint,
        args.cotracker_checkpoint,
        device=args.device,
        mask_score_threshold=args.mask_score_threshold,
    )
    result = extractor.extract(args.video, args.target)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, args.output)
    print(f"Saved object motion to {args.output}: x={tuple(result['x'].shape)}, confidence={tuple(result['confidence'].shape)}")


if __name__ == "__main__":
    main()

