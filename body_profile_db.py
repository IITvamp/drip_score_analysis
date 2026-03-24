"""
body_profile_db.py
------------------
Body-profile retrieval layer backed by Celeb-FBI pose-estimation dataset.

This module builds (or loads) a lightweight local index from:
- alecccdd/celeb-fbi-pose-estimation (pose + biometrics)
- alecccdd/celeb-fbi (image source)

Then it returns nearest body-type matches for a query body vector.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np



@dataclass
class BodyProfileMatch:
    rank: int
    label: str
    summary: str
    similarity_score: float  # 0..1
    source: str  # "history" | "archetype"
    image_path: str | None = None
    celeb_id: str | None = None
    height_cm: float | None = None
    weight_kg: float | None = None
    age: int | None = None
    gender: int | None = None


_CACHE_DIR = Path("cache/celeb_body_profiles")
_INDEX_PATH = _CACHE_DIR / "index.json"
_IMAGE_DIR = _CACHE_DIR / "images"


class BodyProfileDB:
    def __init__(self):
        self._profiles: list[dict[str, Any]] = []
        self._ready = False
        self._demo_max_rows = int(os.getenv("DRIP_CELEB_DEMO_MAX_ROWS", "0") or "0")
        self._index_path = (
            _CACHE_DIR / f"index_demo_{self._demo_max_rows}.json"
            if self._demo_max_rows > 0
            else _INDEX_PATH
        )
        self._load_or_build_index()

    def add_profile(self, body_vector: dict, frame_bgr=None):
        # Intentionally no-op. We only search against public dataset profiles.
        _ = (body_vector, frame_bgr)

    def _distance(self, query: dict, cand: dict) -> float:
        q_shape = str(query.get("body_shape", "unknown"))
        q_sh = float(query.get("shoulder_hip_ratio", 1.0))
        q_tl = float(query.get("torso_leg_ratio", 0.5))
        q_hb = int(query.get("height_bucket", 1))

        c_shape = str(cand.get("body_shape", "unknown"))
        c_sh = float(cand.get("shoulder_hip_ratio", 1.0))
        c_tl = float(cand.get("torso_leg_ratio", 0.5))
        c_hb = int(cand.get("height_bucket", 1))

        # Core proportion similarity.
        d_shape = 0.0 if q_shape == c_shape else 0.22
        d_sh = abs(q_sh - c_sh) * 1.9
        d_tl = abs(q_tl - c_tl) * 1.6
        d_hb = abs(q_hb - c_hb) * 0.14

        # Soft regularizers when available (do not dominate geometry).
        d_bmi = 0.0
        c_bmi = cand.get("bmi")
        if c_bmi is not None:
            d_bmi = min(0.15, abs(float(c_bmi) - 23.0) / 60.0)

        d_age = 0.0
        c_age = cand.get("age")
        if c_age is not None and c_age >= 0:
            d_age = min(0.08, abs(int(c_age) - 32) / 500.0)

        return d_shape + d_sh + d_tl + d_hb + d_bmi + d_age

    def _classify_shape(self, shoulder_hip_ratio: float, torso_leg_ratio: float) -> str:
        if shoulder_hip_ratio > 1.18:
            return "inverted_triangle"
        if shoulder_hip_ratio < 0.88:
            return "triangle"
        if 0.88 <= shoulder_hip_ratio <= 1.12 and torso_leg_ratio < 0.53:
            return "hourglass"
        if torso_leg_ratio > 0.58:
            return "oval"
        return "rectangle"

    def _height_bucket(self, height_cm: float | None) -> int:
        if height_cm is None or height_cm <= 0:
            return 1
        if height_cm < 165:
            return 0
        if height_cm < 180:
            return 1
        return 2

    def _safe_float(self, value: Any, default: float | None = None) -> float | None:
        try:
            if value is None:
                return default
            out = float(value)
            if np.isnan(out):
                return default
            return out
        except (ValueError, TypeError):
            return default

    def _extract_ratio_features(self, row: dict[str, Any]) -> tuple[float, float]:
        # Prefer 3D joints if available.
        joints = row.get("pred_joint_coords")
        shoulder_hip_ratio = None
        torso_leg_ratio = None
        if isinstance(joints, list) and len(joints) > 28:
            try:
                arr = np.asarray(joints, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    l_sh, r_sh = arr[11, :2], arr[12, :2]
                    l_hip, r_hip = arr[23, :2], arr[24, :2]
                    l_ankle, r_ankle = arr[27, :2], arr[28, :2]
                    sh = float(np.linalg.norm(l_sh - r_sh))
                    hip = float(np.linalg.norm(l_hip - r_hip))
                    mid_sh = (l_sh + r_sh) / 2.0
                    mid_hip = (l_hip + r_hip) / 2.0
                    mid_ankle = (l_ankle + r_ankle) / 2.0
                    torso = float(np.linalg.norm(mid_sh - mid_hip))
                    leg = float(np.linalg.norm(mid_hip - mid_ankle))
                    shoulder_hip_ratio = sh / max(hip, 1e-6)
                    torso_leg_ratio = torso / max(torso + leg, 1e-6)
            except Exception:
                shoulder_hip_ratio = None
                torso_leg_ratio = None

        # Fallback using bbox geometry if joints are malformed.
        if shoulder_hip_ratio is None or torso_leg_ratio is None:
            bbox = row.get("bbox")
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = [self._safe_float(v, 0.0) for v in bbox]
                bw = max(1.0, float(x2 - x1))
                bh = max(1.0, float(y2 - y1))
                shoulder_hip_ratio = 1.0
                torso_leg_ratio = float(min(0.7, max(0.38, 0.46 + (bh / (bw * 8.0)))))
            else:
                shoulder_hip_ratio = 1.0
                torso_leg_ratio = 0.5
        return float(shoulder_hip_ratio), float(torso_leg_ratio)

    def _profile_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        shoulder_hip_ratio, torso_leg_ratio = self._extract_ratio_features(row)
        height_cm = self._safe_float(row.get("height"), None)
        if height_cm is not None and height_cm <= 0:
            height_cm = None
        weight_kg = self._safe_float(row.get("weight"), None)
        if weight_kg is not None and weight_kg <= 0:
            weight_kg = None
        age = int(row.get("age", -1)) if row.get("age", -1) not in (None, "",) else -1
        gender = int(row.get("gender", -1)) if row.get("gender", -1) not in (None, "",) else -1

        bmi = None
        if height_cm and height_cm > 0 and weight_kg and weight_kg > 0:
            bmi = float(weight_kg / ((height_cm / 100.0) ** 2))

        body_shape = self._classify_shape(shoulder_hip_ratio, torso_leg_ratio)
        h_bucket = self._height_bucket(height_cm)
        celeb_id = str(row.get("id", "unknown"))

        parts = [f"id={celeb_id}", f"shape={body_shape}", f"sh/hip={shoulder_hip_ratio:.2f}"]
        if height_cm:
            parts.append(f"{height_cm:.0f}cm")
        if weight_kg:
            parts.append(f"{weight_kg:.0f}kg")
        if age >= 0:
            parts.append(f"age {age}")
        summary = " | ".join(parts)

        return {
            "id": celeb_id,
            "label": f"Celeb-FBI #{celeb_id}",
            "summary": summary,
            "body_shape": body_shape,
            "shoulder_hip_ratio": round(shoulder_hip_ratio, 4),
            "torso_leg_ratio": round(torso_leg_ratio, 4),
            "height_bucket": int(h_bucket),
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "age": age if age >= 0 else None,
            "gender": gender if gender >= 0 else None,
            "bmi": bmi,
            "image_path": str(_IMAGE_DIR / f"{celeb_id}.jpg"),
        }

    def _load_or_build_index(self):
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        if self._index_path.exists():
            try:
                self._profiles = json.loads(self._index_path.read_text())
                self._ready = len(self._profiles) > 0
                if self._ready:
                    print(f"Loaded Celeb-FBI body index: {len(self._profiles)} profiles [{self._index_path.name}]")
                    return
            except Exception:
                self._profiles = []

        self._build_index()

    def _build_index(self):
        mode_msg = (
            f"demo mode max_rows={self._demo_max_rows}"
            if self._demo_max_rows > 0
            else "full mode"
        )
        print(f"Building Celeb-FBI body index (one-time download/caching, {mode_msg})...")
        try:
            from datasets import load_dataset

            pose_ds = load_dataset("alecccdd/celeb-fbi-pose-estimation", split="train")
            profiles = []
            for idx, row in enumerate(pose_ds):
                if self._demo_max_rows > 0 and idx >= self._demo_max_rows:
                    break
                try:
                    profiles.append(self._profile_from_row(row))
                except Exception:
                    continue

            # Keep deterministic order for stable ranks.
            profiles.sort(key=lambda p: p.get("id", ""))
            self._profiles = profiles
            self._index_path.write_text(json.dumps(self._profiles, indent=2))
            self._ready = len(self._profiles) > 0
            print(f"Built Celeb-FBI body index: {len(self._profiles)} profiles [{self._index_path.name}]")
        except Exception as e:
            print(f"Failed to build Celeb-FBI index, using fallback archetypes: {e}")
            self._profiles = [
                {
                    "id": "athletic_v",
                    "label": "Athletic V-shape",
                    "summary": "Broad shoulders with balanced lower body.",
                    "body_shape": "inverted_triangle",
                    "shoulder_hip_ratio": 1.22,
                    "torso_leg_ratio": 0.49,
                    "height_bucket": 1,
                    "image_path": None,
                },
                {
                    "id": "balanced_hourglass",
                    "label": "Balanced Hourglass",
                    "summary": "Shoulders and hips are proportionally aligned.",
                    "body_shape": "hourglass",
                    "shoulder_hip_ratio": 1.00,
                    "torso_leg_ratio": 0.50,
                    "height_bucket": 1,
                    "image_path": None,
                },
                {
                    "id": "classic_rectangle",
                    "label": "Classic Rectangle",
                    "summary": "Clean vertical silhouette and even proportions.",
                    "body_shape": "rectangle",
                    "shoulder_hip_ratio": 1.04,
                    "torso_leg_ratio": 0.52,
                    "height_bucket": 1,
                    "image_path": None,
                },
            ]
            self._ready = True

    def _ensure_images(self, celeb_ids: list[str]):
        missing = [cid for cid in celeb_ids if not (_IMAGE_DIR / f"{cid}.jpg").exists()]
        if not missing:
            return
        try:
            from datasets import load_dataset

            # We fetch only train split for now (same id-space coverage in most runs).
            image_ds = load_dataset("alecccdd/celeb-fbi", split="train")
            missing_set = set(missing)
            for row in image_ds:
                sid = str(row.get("id", ""))
                if sid not in missing_set:
                    continue
                img = row.get("image")
                if img is None:
                    continue
                out = _IMAGE_DIR / f"{sid}.jpg"
                img.convert("RGB").save(out, "JPEG", quality=90)
                missing_set.remove(sid)
                if not missing_set:
                    break
        except Exception as e:
            print(f"Could not cache Celeb-FBI images: {e}")

    def search(
        self,
        body_vector: dict | None,
        top_k: int = 8,
        exclude_image_path: str | None = None,
        query_gender: int | None = None,
        strict_gender: bool = True,
    ) -> list[BodyProfileMatch]:
        if not body_vector or not self._ready:
            return []

        scored = []
        qg: int | None = None
        if strict_gender and query_gender in (0, 1):
            qg = int(query_gender)
        elif strict_gender and query_gender is not None:
            # invalid gender value; ignore filter
            qg = None
        for cand in self._profiles:
            if qg is not None:
                cg = cand.get("gender")
                if cg is None:
                    continue  # strict mode: exclude unknown gender
                if int(cg) != qg:
                    continue
            d = self._distance(body_vector, cand)
            sim = 1.0 / (1.0 + d)
            scored.append((sim, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        selected = scored[:top_k]
        celeb_ids = [str(cand.get("id", "")) for _, cand in selected if str(cand.get("id", "")).isdigit()]
        self._ensure_images(celeb_ids)

        out: list[BodyProfileMatch] = []
        for i, (sim, cand) in enumerate(selected, start=1):
            image_path = cand.get("image_path")
            if image_path and not Path(image_path).exists():
                image_path = None
            if exclude_image_path and image_path and Path(image_path).resolve() == Path(exclude_image_path).resolve():
                continue
            out.append(
                BodyProfileMatch(
                    rank=i,
                    label=str(cand.get("label", "Profile")),
                    summary=str(cand.get("summary", "")),
                    similarity_score=float(max(0.0, min(1.0, sim))),
                    source="celeb_fbi",
                    image_path=image_path,
                    celeb_id=str(cand.get("id", "")),
                    height_cm=cand.get("height_cm"),
                    weight_kg=cand.get("weight_kg"),
                    age=cand.get("age"),
                    gender=cand.get("gender"),
                )
            )
        return out
