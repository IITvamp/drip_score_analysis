"""
prefetch_celeb_images.py
------------------------
Download and cache Celeb-FBI images locally for faster body-profile previews.

Usage:
    python prefetch_celeb_images.py
    python prefetch_celeb_images.py --limit 500
"""

from __future__ import annotations

import argparse
from pathlib import Path


def prefetch(limit: int | None = None):
    from datasets import load_dataset

    out_dir = Path("cache/celeb_body_profiles/images")
    out_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    total_seen = 0

    for split in ("train", "test"):
        ds = load_dataset("alecccdd/celeb-fbi", split=split)
        for row in ds:
            sid = str(row.get("id", "")).strip()
            if not sid:
                continue
            total_seen += 1

            out_path = out_dir / f"{sid}.jpg"
            if out_path.exists():
                continue

            img = row.get("image")
            if img is None:
                continue

            img.convert("RGB").save(out_path, "JPEG", quality=90)
            total_saved += 1

            if limit is not None and total_saved >= limit:
                print(
                    f"Prefetch complete (limit reached). "
                    f"saved={total_saved}, seen={total_seen}, dir={out_dir}"
                )
                return

    print(f"Prefetch complete. saved={total_saved}, seen={total_seen}, dir={out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefetch Celeb-FBI images into local cache")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of new images to save (for quick testing)",
    )
    args = parser.parse_args()
    prefetch(limit=args.limit)

