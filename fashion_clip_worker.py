"""
fashion_clip_worker.py
----------------------
Isolated embedding worker for FashionCLIP.

Runs in a dedicated Python environment without mediapipe/opencv.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def _to_feature_tensor(out):
    if torch.is_tensor(out):
        return out
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.mean(dim=1)
    raise RuntimeError("Unsupported CLIP output type")


def main():
    parser = argparse.ArgumentParser(description="FashionCLIP embedding worker")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", required=True, help="Output .npy path")
    parser.add_argument(
        "--model-id",
        default="patrickjohncyh/fashion-clip",
        help="Hugging Face model id",
    )
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    img = Image.open(args.image).convert("RGB")
    processor = CLIPProcessor.from_pretrained(args.model_id, use_fast=False)
    model = CLIPModel.from_pretrained(args.model_id)
    model.eval()

    with torch.no_grad():
        inputs = processor(images=img, return_tensors="pt")
        feats = _to_feature_tensor(model.get_image_features(**inputs))
        feats = feats / feats.norm(dim=-1, keepdim=True)

    np.save(args.output, feats.squeeze(0).cpu().numpy().astype(np.float32))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
