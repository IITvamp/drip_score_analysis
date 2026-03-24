"""
dataset_builder.py
------------------
Downloads Marqo/deepfashion-multimodal from HuggingFace, embeds all images
using Marqo-FashionSigLIP, and builds a FAISS index for fast similarity search.

Usage:
    python dataset_builder.py                 # full build (42k images, ~2-3 hrs)
    python dataset_builder.py --max 2000      # quick build with 2000 samples
    python dataset_builder.py --max 500       # demo build (5-10 min)

Outputs:
    cache/faiss_index.bin       FAISS flat cosine index
    cache/metadata.json         item_ID, category, text description, image URL
    cache/build_info.json       build stats
"""

import os
import json
import argparse
import time
import numpy as np
from tqdm import tqdm
from pathlib import Path

CACHE_DIR = Path("cache")
INDEX_PATH = CACHE_DIR / "faiss_index.bin"
META_PATH = CACHE_DIR / "metadata.json"
INFO_PATH = CACHE_DIR / "build_info.json"

BATCH_SIZE = 64


def _resolve_backend(backend: str | None) -> str:
    selected = (backend or os.getenv("DRIP_EMBED_BACKEND", "hf_clip")).strip().lower()
    aliases = {
        "marqo": "marqo_siglip",
        "marqo_fashionsiglip": "marqo_siglip",
        "siglip": "marqo_siglip",
        "fashionclip": "fashion_clip",
    }
    selected = aliases.get(selected, selected)
    if selected not in {"hf_clip", "marqo_siglip", "fashion_clip"}:
        return "hf_clip"
    return selected


def _backend_model_id(selected: str) -> str:
    if selected == "fashion_clip":
        return "patrickjohncyh/fashion-clip"
    return "openai/clip-vit-base-patch32"


def build_dataset(max_items: int | None = None, backend: str | None = None):
    import torch
    import faiss
    from datasets import load_dataset
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    CACHE_DIR.mkdir(exist_ok=True)
    selected = _resolve_backend(backend)

    # ---- Load model ----
    if selected == "marqo_siglip":
        import open_clip

        print("Loading Marqo-FashionSigLIP...")
        model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        model.eval()
        processor = None
        model_name = "Marqo/marqo-fashionSigLIP"
    else:
        model_name = _backend_model_id(selected)
        print(f"Loading HF CLIP ({model_name})...")
        processor = CLIPProcessor.from_pretrained(model_name, use_fast=False)
        model = CLIPModel.from_pretrained(model_name)
        model.eval()
        preprocess = None
    print("Model loaded.\n")

    # ---- Load dataset (streaming to avoid full download) ----
    print("Connecting to HuggingFace dataset (Marqo/deepfashion-multimodal)...")
    ds = load_dataset(
        "Marqo/deepfashion-multimodal",
        split="data",
        streaming=True,
    )

    # ---- Embed in batches ----
    embeddings = []
    metadata = []
    batch_images = []
    batch_meta = []

    total = max_items or 42537
    pbar = tqdm(total=min(total, 42537), desc="Embedding images", unit="img")
    count = 0
    errors = 0

    def flush_batch():
        nonlocal errors
        if not batch_images:
            return
        try:
            with torch.no_grad():
                if selected == "marqo_siglip":
                    tensors = torch.stack([preprocess(img) for img in batch_images])
                    feats = model.encode_image(tensors)
                else:
                    inputs = processor(images=batch_images, return_tensors="pt")
                    feats = model.get_image_features(**inputs)
                    if not torch.is_tensor(feats):
                        if hasattr(feats, "pooler_output") and feats.pooler_output is not None:
                            feats = feats.pooler_output
                        elif hasattr(feats, "last_hidden_state"):
                            feats = feats.last_hidden_state.mean(dim=1)
                        else:
                            raise RuntimeError("Unsupported CLIP output type from get_image_features")
                feats = feats / feats.norm(dim=-1, keepdim=True)
            arr = feats.cpu().numpy().astype(np.float32)
            for i, meta in enumerate(batch_meta):
                embeddings.append(arr[i])
                metadata.append(meta)
        except Exception as e:
            print(f"Batch embedding error: {e}")
            errors += len(batch_images)
        batch_images.clear()
        batch_meta.clear()

    for item in ds:
        if max_items and count >= max_items:
            break

        try:
            img = item["image"]
            if img is None:
                continue
            if not isinstance(img, Image.Image):
                img = Image.fromarray(np.array(img))
            img = img.convert("RGB")

            meta = {
                "item_id": str(item.get("item_ID", f"item_{count}")),
                "category1": str(item.get("category1", "")),
                "category2": str(item.get("category2", "")),
                "text": str(item.get("text", "")),
                "index": count,
            }

            batch_images.append(img)
            batch_meta.append(meta)
            count += 1
            pbar.update(1)

            if len(batch_images) >= BATCH_SIZE:
                flush_batch()

        except Exception:
            errors += 1
            continue

    flush_batch()
    pbar.close()

    if not embeddings:
        print("ERROR: No embeddings generated. Check your internet connection.")
        return

    # ---- Build FAISS index ----
    print(f"\nBuilding FAISS index from {len(embeddings)} embeddings...")
    dim = embeddings[0].shape[0]
    matrix = np.stack(embeddings)

    # Normalize for cosine similarity via inner product
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-8)

    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    # ---- Save ----
    faiss.write_index(index, str(INDEX_PATH))
    with open(META_PATH, "w") as f:
        json.dump(metadata, f)

    info = {
        "total_items": len(embeddings),
        "embedding_dim": dim,
        "errors": errors,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "backend": selected,
        "model": model_name,
        "dataset": "Marqo/deepfashion-multimodal",
    }
    with open(INFO_PATH, "w") as f:
        json.dump(info, f, indent=2)

    print(f"\nDone!")
    print(f"  Items indexed : {len(embeddings)}")
    print(f"  Errors skipped: {errors}")
    print(f"  Embedding dim : {dim}")
    print(f"  FAISS index   : {INDEX_PATH}")
    print(f"  Metadata      : {META_PATH}")


def is_index_built() -> bool:
    return INDEX_PATH.exists() and META_PATH.exists()


def get_build_info() -> dict | None:
    if INFO_PATH.exists():
        with open(INFO_PATH) as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build DeepFashion FAISS index")
    parser.add_argument("--max", type=int, default=None,
                        help="Max items to index (default: all 42k)")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        choices=["hf_clip", "fashion_clip", "marqo_siglip"],
        help="Embedding backend to build FAISS index with",
    )
    args = parser.parse_args()
    build_dataset(max_items=args.max, backend=args.backend)
