"""
vector_db.py
------------
Loads the pre-built FAISS index and metadata, exposes a search() method
that takes a fashion embedding and returns the top-K most similar outfit
items from the DeepFashion dataset, sorted by similarity score descending.

If the index has not been built yet, it falls back to a guided message.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field

CACHE_DIR = Path("cache")
INDEX_PATH = CACHE_DIR / "faiss_index.bin"
META_PATH = CACHE_DIR / "metadata.json"


@dataclass
class FashionMatch:
    rank: int
    item_id: str
    category1: str          # men / women
    category2: str          # denim, jackets, etc.
    description: str        # natural language outfit description
    similarity_score: float # cosine similarity 0–1
    image: object = field(default=None, repr=False)  # PIL Image if fetched


class FashionVectorDB:
    def __init__(self):
        self._index = None
        self._metadata: list[dict] = []
        self._loaded = False

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------

    def load(self) -> bool:
        """Load FAISS index and metadata. Returns True if successful."""
        if self._loaded:
            return True

        if not INDEX_PATH.exists() or not META_PATH.exists():
            return False

        try:
            import faiss
            self._index = faiss.read_index(str(INDEX_PATH))
            with open(META_PATH) as f:
                self._metadata = json.load(f)
            self._loaded = True
            print(f"Loaded FAISS index: {self._index.ntotal} items")
            return True
        except Exception as e:
            print(f"Failed to load index: {e}")
            return False

    def is_ready(self) -> bool:
        return self._loaded and self._index is not None

    def item_count(self) -> int:
        return self._index.ntotal if self._index else 0

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query_embedding: np.ndarray, top_k: int = 10,
               category_filter: str | None = None) -> list[FashionMatch]:
        """
        Search for the most similar fashion items.

        Args:
            query_embedding : L2-normalised float32 array from FashionSigLIP
            top_k           : number of results to return
            category_filter : 'men' or 'women' (optional)

        Returns:
            List of FashionMatch sorted by similarity_score descending.
        """
        if not self.is_ready():
            return []

        # Normalize query
        q = query_embedding.astype(np.float32).reshape(1, -1)
        if self._index is not None and q.shape[1] != self._index.d:
            print(
                f"Embedding dimension mismatch: query={q.shape[1]} index={self._index.d}. "
                "Rebuild dataset index with matching embedding backend."
            )
            return []
        q = q / (np.linalg.norm(q) + 1e-8)

        # Fetch more candidates if filtering by category
        fetch_k = top_k * 5 if category_filter else top_k * 2
        fetch_k = min(fetch_k, self._index.ntotal)

        scores, indices = self._index.search(q, fetch_k)
        scores = scores[0]
        indices = indices[0]

        results = []
        for rank_raw, (score, idx) in enumerate(zip(scores, indices)):
            if idx < 0 or idx >= len(self._metadata):
                continue

            meta = self._metadata[idx]

            # Apply category filter
            if category_filter:
                if meta.get("category1", "").lower() != category_filter.lower():
                    continue

            results.append(FashionMatch(
                rank=len(results) + 1,
                item_id=meta.get("item_id", ""),
                category1=meta.get("category1", ""),
                category2=meta.get("category2", ""),
                description=meta.get("text", ""),
                similarity_score=float(np.clip(score, 0, 1)),
            ))

            if len(results) >= top_k:
                break

        return results

    # ------------------------------------------------------------------
    # Fetch image for a result (lazy)
    # ------------------------------------------------------------------

    def fetch_image(self, match: FashionMatch, size: tuple = (224, 224)):
        """
        Fetch the actual image from HuggingFace dataset for display.
        Returns PIL Image or None. Uses streaming so only this item is downloaded.
        """
        try:
            from datasets import load_dataset
            from PIL import Image

            # Parse item index from metadata
            idx = None
            for i, m in enumerate(self._metadata):
                if m.get("item_id") == match.item_id:
                    idx = m.get("index", i)
                    break

            if idx is None:
                return None

            ds = load_dataset(
                "Marqo/deepfashion-multimodal",
                split="data",
                streaming=True,
            )
            for i, item in enumerate(ds):
                if i == idx:
                    img = item.get("image")
                    if img is not None:
                        if not isinstance(img, Image.Image):
                            import numpy as np
                            img = Image.fromarray(np.array(img))
                        img = img.convert("RGB").resize(size)
                        return img
                    break
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    # Batch image fetch for display panel
    # ------------------------------------------------------------------

    def fetch_images_batch(self, matches: list[FashionMatch],
                           size: tuple = (224, 224)) -> dict[str, object]:
        """
        Fetch images for a list of matches.
        Returns dict {item_id: PIL Image}.
        Streams through dataset once, collecting all needed indices.
        """
        from datasets import load_dataset
        from PIL import Image
        import numpy as np

        # Build target index → item_id map
        target_map = {}
        for match in matches:
            for m in self._metadata:
                if m.get("item_id") == match.item_id:
                    target_map[m.get("index", -1)] = match.item_id
                    break

        if not target_map:
            return {}

        max_idx = max(target_map.keys())
        result = {}

        try:
            ds = load_dataset(
                "Marqo/deepfashion-multimodal",
                split="data",
                streaming=True,
            )
            for i, item in enumerate(ds):
                if i > max_idx + 10:
                    break
                if i in target_map:
                    img = item.get("image")
                    if img is not None:
                        if not isinstance(img, Image.Image):
                            img = Image.fromarray(np.array(img))
                        img = img.convert("RGB").resize(size)
                        result[target_map[i]] = img
                    if len(result) >= len(target_map):
                        break
        except Exception as e:
            print(f"Image fetch error: {e}")

        return result
