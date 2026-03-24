"""
app.py
------
Main Drip Score application.

Controls:
  S  — scan drip (capture & analyze)
  Q  — quit
  D  — toggle debug overlay
  R  — reset / back to live feed
  1-9 — scroll through fashion matches

Run:
    python app.py

First run: will prompt you to build the dataset index.
"""

import sys
import os
import time
import argparse
import json
import cv2
import numpy as np
from pathlib import Path

# Local modules
from mediapipe_compat import get_mediapipe_solutions
from normalizer import PerspectiveNormalizer
from feature_extractor import (
    extract_body_vector,
    extract_skin_tone,
    extract_clothing_colors,
    extract_fashion_embedding,
    load_fashion_model,
)
from scorer import calculate_drip_score
from body_profile_db import BodyProfileDB
from vector_db import FashionVectorDB
from commentary import get_commentary, score_label, score_color_bgr

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------

WIN_W, WIN_H = 1280, 720
RESULT_COLS = 4          # fashion match thumbnails per row
THUMB_W, THUMB_H = 200, 220

# -----------------------------------------------------------------------
# Helper: draw text with background
# -----------------------------------------------------------------------

def put_text(img, text, x, y, scale=0.55, color=(255, 255, 255),
             thickness=1, bg=True, bg_color=(0, 0, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    if bg:
        cv2.rectangle(img, (x - 3, y - th - 3), (x + tw + 3, y + baseline + 2),
                      bg_color, -1)
    cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_bar(img, x, y, w, h, value, max_val, color):
    fill_w = int(w * value / max(max_val, 1))
    cv2.rectangle(img, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(img, (x, y), (x + fill_w, y + h), color, -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (120, 120, 120), 1)


def pil_to_cv(pil_img):
    import numpy as np
    arr = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


# -----------------------------------------------------------------------
# Result panel builder
# -----------------------------------------------------------------------

def build_result_panel(frame_bgr, result, matches, match_images,
                       debug=False, match_scroll=0) -> np.ndarray:
    """
    Compose a 1280×720 result panel:
    - Left 420px : frozen frame + score overlay
    - Right 860px: score breakdown + fashion matches
    """
    canvas = np.zeros((WIN_H, WIN_W, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)

    score = result["total"]
    s_color = score_color_bgr(score)

    # ---- LEFT: frozen frame ----
    frame_h, frame_w = frame_bgr.shape[:2]
    scale = min(400 / frame_w, 680 / frame_h)
    fw = int(frame_w * scale)
    fh = int(frame_h * scale)
    frame_small = cv2.resize(frame_bgr, (fw, fh))
    fx = (420 - fw) // 2
    fy = (WIN_H - fh) // 2
    canvas[fy:fy + fh, fx:fx + fw] = frame_small

    # Score badge on frame
    badge_text = f"{score}"
    cv2.rectangle(canvas, (fx, fy), (fx + 90, fy + 60), s_color, -1)
    cv2.putText(canvas, badge_text, (fx + 6, fy + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)

    label = score_label(score)
    put_text(canvas, label, fx, fy + 76, scale=0.55, color=s_color, bg=False)

    # Body shape + skin season
    put_text(canvas, f"Shape: {result['body_shape']}", fx, fy + 100,
             scale=0.45, color=(200, 200, 200), bg=False)
    put_text(canvas, f"Season: {result['skin_season']}", fx, fy + 122,
             scale=0.45, color=(200, 200, 200), bg=False)

    # ---- DIVIDER ----
    cv2.line(canvas, (428, 20), (428, WIN_H - 20), (60, 60, 60), 1)

    # ---- RIGHT: score breakdown ----
    rx = 440
    ry = 30

    put_text(canvas, "DRIP SCORE BREAKDOWN", rx, ry + 20,
             scale=0.65, color=(220, 220, 220), thickness=2, bg=False)

    pillars = [
        ("Fit & proportion", result["fit_proportion"],   20, result["breakdown"]["fit"]),
        ("Color harmony",    result["color_harmony"],    20, result["breakdown"]["harmony"]),
        ("Color season",     result["color_season"],     15, result["breakdown"]["season"]),
        ("Grooming",         result["grooming"],         15, result["breakdown"]["grooming"]),
        ("Style coherence",  result["style_coherence"],  10, result["breakdown"]["coherence"]),
        ("Fashion matches",  result["fashion_retrieval"], 20, result["breakdown"]["fashion_retrieval"]),
    ]

    py = ry + 50
    for name, val, max_v, tip in pillars:
        ratio = val / max_v
        if ratio >= 0.8:
            bar_color = (0, 200, 80)
        elif ratio >= 0.55:
            bar_color = (0, 180, 230)
        else:
            bar_color = (0, 100, 220)

        put_text(canvas, f"{name}", rx, py,
                 scale=0.48, color=(200, 200, 200), bg=False)
        put_text(canvas, f"{val}/{max_v}", rx + 310, py,
                 scale=0.48, color=bar_color, bg=False)
        draw_bar(canvas, rx, py + 4, 300, 10, val, max_v, bar_color)

        # Tip text (truncated)
        tip_short = tip[:68] + "…" if len(tip) > 68 else tip
        put_text(canvas, tip_short, rx, py + 20,
                 scale=0.36, color=(140, 140, 140), bg=False)
        py += 52

    # Commentary lines
    commentary = get_commentary(score, result["body_shape"], result["skin_season"])
    put_text(canvas, f'"{commentary["main_line"]}"', rx, py + 10,
             scale=0.5, color=(255, 220, 60), bg=False)
    put_text(canvas, commentary["shape_line"], rx, py + 34,
             scale=0.42, color=(180, 220, 255), bg=False)
    put_text(canvas, f"Color tip: {commentary['color_tip']}", rx, py + 56,
             scale=0.4, color=(160, 255, 160), bg=False)

    # ---- SIMILAR MATCHES ----
    my = py + 80
    cv2.line(canvas, (rx, my), (WIN_W - 20, my), (60, 60, 60), 1)
    my += 14

    visible_matches = matches[match_scroll: match_scroll + RESULT_COLS]
    outfit_mode = bool(visible_matches) and getattr(visible_matches[0], "item_id", None) is not None
    title = (
        "SIMILAR OUTFIT INSPIRATIONS  (DeepFashion)"
        if outfit_mode else
        "SIMILAR BODY PROFILES  (closest proportion matches)"
    )
    put_text(canvas, title, rx, my, scale=0.48, color=(200, 200, 200), thickness=1, bg=False)
    my += 22

    # Show 4 cards at a time, scroll with 1-9 keys
    tx = rx
    for i, match in enumerate(visible_matches):
        if outfit_mode:
            card_w = THUMB_W - 10
            card_h = THUMB_H - 40
            item_img = match_images.get(match.item_id)
            if item_img is not None:
                thumb = pil_to_cv(item_img)
                thumb = cv2.resize(thumb, (card_w, card_h))
                canvas[my:my + card_h, tx:tx + card_w] = thumb
            else:
                cv2.rectangle(canvas, (tx, my), (tx + card_w, my + card_h), (40, 40, 50), -1)
                put_text(canvas, "Loading...", tx + 20, my + 80,
                         scale=0.4, color=(120, 120, 120), bg=False)

            sim_pct = int(match.similarity_score * 100)
            sim_color = (0, 200, 80) if sim_pct > 75 else (0, 180, 230) if sim_pct > 55 else (100, 180, 255)
            put_text(canvas, f"{sim_pct}%", tx + 2, my + card_h - 4,
                     scale=0.45, color=sim_color, bg=True, bg_color=(0, 0, 0))
            put_text(canvas, f"#{match.rank} {match.category2[:14]}", tx + 2, my + 16,
                     scale=0.34, color=(220, 220, 220), bg=True, bg_color=(0, 0, 0))
            desc = (match.description[:34] + "…") if len(match.description) > 34 else match.description
            put_text(canvas, desc, tx + 2, my + 34, scale=0.30, color=(180, 220, 180), bg=True, bg_color=(0, 0, 0))
            tx += THUMB_W + 4
            if tx + THUMB_W > WIN_W:
                break
            continue

        card_w = THUMB_W - 10
        card_h = THUMB_H - 40
        cv2.rectangle(canvas, (tx, my), (tx + card_w, my + card_h), (34, 38, 46), -1)
        cv2.rectangle(canvas, (tx, my), (tx + card_w, my + card_h), (70, 80, 98), 1)

        # Optional thumbnail from stored history profile
        thumb_h = 92
        image_path = getattr(match, "image_path", None)
        if image_path and Path(image_path).exists():
            thumb = cv2.imread(image_path)
            if thumb is not None:
                thumb = cv2.resize(thumb, (card_w - 8, thumb_h))
                canvas[my + 4: my + 4 + thumb_h, tx + 4: tx + 4 + card_w - 8] = thumb
                cv2.rectangle(canvas, (tx + 4, my + 4), (tx + 4 + card_w - 8, my + 4 + thumb_h), (55, 55, 60), 1)

        # Score badge
        sim_pct = int(match.similarity_score * 100)
        sim_color = (0, 200, 80) if sim_pct > 75 else (0, 180, 230) if sim_pct > 55 else (100, 180, 255)
        put_text(canvas, f"{sim_pct}%", tx + 2, my + card_h - 4,
                 scale=0.45, color=sim_color, bg=True, bg_color=(0, 0, 0))

        raw_source = str(getattr(match, "source", ""))
        if raw_source == "history":
            source = "history"
        elif raw_source == "celeb_fbi":
            source = "celeb-fbi"
        else:
            source = "archetype"
        meta_y = my + (thumb_h + 18 if image_path else 18)
        put_text(canvas, f"#{match.rank}  {source}", tx + 6, meta_y,
                 scale=0.38, color=(150, 170, 190), bg=False)

        label = getattr(match, "label", "Body profile")
        desc = getattr(match, "summary", "")
        put_text(canvas, label[:24], tx + 6, meta_y + 24, scale=0.40, color=(220, 220, 220), bg=False)
        put_text(canvas, (desc[:44] + "…") if len(desc) > 44 else desc,
                 tx + 6, meta_y + 44, scale=0.32, color=(160, 190, 160), bg=False)

        tx += THUMB_W + 4
        if tx + THUMB_W > WIN_W:
            break

    # Navigation hint
    if len(matches) > RESULT_COLS:
        total_pages = (len(matches) + RESULT_COLS - 1) // RESULT_COLS
        cur_page = match_scroll // RESULT_COLS + 1
        nav_y = min(WIN_H - 20, my + THUMB_H + 10)
        put_text(canvas, f"Press 1-9 to page  |  Page {cur_page}/{total_pages}",
                 rx, nav_y, scale=0.4, color=(140, 140, 140), bg=False)

    put_text(canvas, "Press R to retake  |  Q to quit", WIN_W - 280, WIN_H - 16,
             scale=0.4, color=(100, 100, 100), bg=False)

    return canvas


# -----------------------------------------------------------------------
# Live feed overlay
# -----------------------------------------------------------------------

def draw_live_overlay(frame: np.ndarray, pose_results, normalizer: PerspectiveNormalizer,
                      debug: bool = False) -> np.ndarray:
    h, w = frame.shape[:2]
    overlay = frame.copy()

    # Person detected indicators
    if pose_results and pose_results.pose_landmarks:
        lm = pose_results.pose_landmarks.landmark
        ls = (int(lm[11].x * w), int(lm[11].y * h))
        rs = (int(lm[12].x * w), int(lm[12].y * h))
        lh = (int(lm[23].x * w), int(lm[23].y * h))
        rh = (int(lm[24].x * w), int(lm[24].y * h))

        # Shoulder line
        cv2.line(overlay, ls, rs, (0, 220, 80), 2)
        # Hip line
        cv2.line(overlay, lh, rh, (0, 180, 255), 2)
        # Shoulder to hip
        cv2.line(overlay, ((ls[0]+rs[0])//2, (ls[1]+rs[1])//2),
                 ((lh[0]+rh[0])//2, (lh[1]+rh[1])//2), (80, 200, 255), 1)

        if normalizer.is_valid():
            put_text(overlay, "Person detected — Press S to scan",
                     20, h - 30, scale=0.6, color=(0, 220, 80), bg=True, bg_color=(0, 0, 0))
        else:
            put_text(overlay, "Move closer — face not detected",
                     20, h - 30, scale=0.6, color=(0, 160, 255), bg=True)
    else:
        put_text(overlay, "Stand in full view — full body needed",
                 20, h - 30, scale=0.6, color=(0, 100, 255), bg=True)

    if debug and normalizer.is_valid():
        info = normalizer.debug_info()
        put_text(overlay, f"FaceUnit: {info['face_unit_px']:.1f}px  ({info['estimated_distance']})",
                 20, 40, scale=0.5, color=(255, 220, 0), bg=True)

    put_text(overlay, "S=Scan  D=Debug  Q=Quit", w - 200, 30,
             scale=0.45, color=(160, 160, 160), bg=True, bg_color=(0, 0, 0))

    return cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)


# -----------------------------------------------------------------------
# Loading screen
# -----------------------------------------------------------------------

def loading_screen(msg: str, w=WIN_W, h=WIN_H) -> np.ndarray:
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)
    put_text(canvas, "DRIP SCORE", w // 2 - 100, h // 2 - 60,
             scale=1.4, color=(220, 200, 60), thickness=3, bg=False)
    put_text(canvas, msg, w // 2 - 220, h // 2 + 10,
             scale=0.65, color=(200, 200, 200), bg=False)
    dots = "." * (int(time.time() * 2) % 4)
    put_text(canvas, dots, w // 2 + 220, h // 2 + 10,
             scale=0.65, color=(200, 200, 200), bg=False)
    return canvas


# -----------------------------------------------------------------------
# Main app
# -----------------------------------------------------------------------

class DripScoreApp:
    def __init__(self):
        self.debug = False
        self.state = "live"          # live | analyzing | results
        self.frozen_frame = None
        self.result = None
        self.matches = []
        self.match_images = {}
        self.outfit_matches = []
        self.outfit_images = {}
        self.match_scroll = 0
        self.normalizer = PerspectiveNormalizer()
        self.body_db = BodyProfileDB()
        self.db = FashionVectorDB()
        self.last_body_vector = None
        self.last_skin_data = None
        self.last_clothing_data = None
        self.last_fashion_embedding = None
        self.embedding_backend = os.getenv("DRIP_EMBED_BACKEND", "hf_clip").strip().lower()
        self.enable_fashion_model = os.getenv("DRIP_ENABLE_FASHION_MODEL", "1") == "1"
        self.query_gender = None
        env_gender = os.getenv("DRIP_QUERY_GENDER", "").strip().lower()
        if env_gender != "":
            if env_gender in {"0", "male", "m"}:
                self.query_gender = 0
            elif env_gender in {"1", "female", "f"}:
                self.query_gender = 1
            else:
                print(f"Warning: DRIP_QUERY_GENDER='{env_gender}' not recognized; disabling gender filter")

        # MediaPipe
        mp_solutions = get_mediapipe_solutions()
        mp_pose = mp_solutions.pose
        mp_face = mp_solutions.face_mesh
        mp_seg = mp_solutions.selfie_segmentation

        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.selfie_seg = mp_seg.SelfieSegmentation(model_selection=1)
        # Better settings for uploaded single-image mode.
        self.pose_static = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.face_mesh_static = mp_face.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self.selfie_seg_static = mp_seg.SelfieSegmentation(model_selection=1)
    def _save_body_match_list(self, body_matches):
        out = []
        for m in body_matches:
            out.append({
                "rank": int(m.rank),
                "celeb_id": str(getattr(m, "celeb_id", "")),
                "label": str(getattr(m, "label", "")),
                "source": str(getattr(m, "source", "")),
                "similarity_score": float(getattr(m, "similarity_score", 0.0)),
                "image_path": getattr(m, "image_path", None),
                "gender": getattr(m, "gender", None),
                "height_cm": getattr(m, "height_cm", None),
                "weight_kg": getattr(m, "weight_kg", None),
                "age": getattr(m, "age", None),
                "summary": str(getattr(m, "summary", "")),
            })
        out_path = Path("cache/body_profile_matches.json")
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
        print(f"Saved body match list to: {out_path}")

    def _print_body_match_list(self, body_matches):
        if not body_matches:
            print("Body profile matches: none")
            return
        print("\nBody profile matches (primary logic):")
        for m in body_matches:
            print(
                f"  #{m.rank}  {m.label}  "
                f"source={m.source}  sim={m.similarity_score:.3f}  "
                f"image_path={getattr(m, 'image_path', None)}"
            )

    def _save_outfit_match_assets(self, outfit_matches, outfit_images):
        out_dir = Path("cache/outfit_matches")
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for m in outfit_matches:
            local_path = None
            img = outfit_images.get(m.item_id)
            if img is not None:
                fname = f"{m.rank:02d}_{m.item_id}.jpg"
                save_path = out_dir / fname
                img.convert("RGB").save(save_path)
                local_path = str(save_path)
            manifest.append({
                "rank": int(m.rank),
                "item_id": str(m.item_id),
                "category1": str(m.category1),
                "category2": str(m.category2),
                "similarity_score": float(m.similarity_score),
                "description": str(m.description),
                "image_path": local_path,
            })
        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"Saved outfit thumbnail manifest to: {manifest_path}")

    def _save_analysis_report(self, image_path: str | None = None):
        """
        Save a unified JSON report for easier downstream visualization.
        """
        if not self.result:
            return

        # Expand body-profile matches into a structured list.
        body_profile_matches = []
        for m in getattr(self, "body_matches", []) or []:
            body_profile_matches.append({
                "rank": int(m.rank),
                "celeb_id": getattr(m, "celeb_id", None),
                "label": str(getattr(m, "label", "")),
                "source": str(getattr(m, "source", "")),
                "similarity_score": float(getattr(m, "similarity_score", 0.0)),
                "image_path": getattr(m, "image_path", None),
                "gender": getattr(m, "gender", None),
                "height_cm": getattr(m, "height_cm", None),
                "weight_kg": getattr(m, "weight_kg", None),
                "age": getattr(m, "age", None),
                "summary": str(getattr(m, "summary", "")),
            })

        # Expand fashion matches similarly.
        fashion_matches = []
        for m in getattr(self, "outfit_matches", []) or []:
            fashion_matches.append({
                "rank": int(m.rank),
                "item_id": str(getattr(m, "item_id", "")),
                "category1": str(getattr(m, "category1", "")),
                "category2": str(getattr(m, "category2", "")),
                "description": str(getattr(m, "description", "")),
                "similarity_score": float(getattr(m, "similarity_score", 0.0)),
                "image_path": str(Path("cache/outfit_matches") / f"{int(m.rank):02d}_{m.item_id}.jpg"),
            })

        score_max = {
            "fit_proportion": 20,
            "color_harmony": 20,
            "color_season": 15,
            "grooming": 15,
            "style_coherence": 10,
            "fashion_retrieval": 20,
            "total": 100,
        }

        report = {
            "version": "1.0",
            "input": {
                "image_path": image_path,
                "embedding_backend": self.embedding_backend,
                "fashion_model_enabled": bool(self.enable_fashion_model),
                "query_gender": self.query_gender,
            },
            "body_profile": self.last_body_vector,
            "skin_profile": self.last_skin_data,
            "clothing_profile": self.last_clothing_data,
            "score": {
                "max": score_max,
                "values": {
                    "fit_proportion": int(self.result.get("fit_proportion", 0)),
                    "color_harmony": int(self.result.get("color_harmony", 0)),
                    "color_season": int(self.result.get("color_season", 0)),
                    "grooming": int(self.result.get("grooming", 0)),
                    "style_coherence": int(self.result.get("style_coherence", 0)),
                    "fashion_retrieval": int(self.result.get("fashion_retrieval", 0)),
                    "total": int(self.result.get("total", 0)),
                },
                "explanations": self.result.get("breakdown", {}),
                "diagnostics": {
                    "outfit_match_delta": self.result.get("outfit_match_delta", {}),
                },
                "calculation": self.result.get("score_calculation", {}),
                "labels": {
                    "score_band": score_label(int(self.result.get("total", 0))),
                    "body_shape": self.result.get("body_shape", "unknown"),
                    "skin_season": self.result.get("skin_season", "unknown"),
                },
            },
            "similar_body_profiles": body_profile_matches,
            "fashion_matches": fashion_matches,
            "artifacts": {
                "result_panel_image": (
                    str(Path("cache") / f"result_{Path(image_path).stem}.jpg")
                    if image_path else None
                ),
                "body_matches_json": "cache/body_profile_matches.json",
                "outfit_manifest_json": "cache/outfit_matches/manifest.json",
            },
            "improvements": self.result.get("improvements", []),
            "why_this_score": str(self.result.get("why_this_score", "")),
        }

        out_name = f"report_{Path(image_path).stem}.json" if image_path else "report_latest.json"
        out_path = Path("cache") / out_name
        out_path.parent.mkdir(exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"Saved analysis report to: {out_path}")

    def _enhance_for_detection(self, frame_bgr: np.ndarray) -> np.ndarray:
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y2 = clahe.apply(y)
        return cv2.cvtColor(cv2.merge([y2, cr, cb]), cv2.COLOR_YCrCb2BGR)

    def _run_static_with_retries(self, frame_bgr: np.ndarray):
        h, w = frame_bgr.shape[:2]
        variants = [
            frame_bgr,
            self._enhance_for_detection(frame_bgr),
            cv2.resize(frame_bgr, (int(w * 1.2), int(h * 1.2))),
        ]
        best = None
        best_score = -1
        for v in variants:
            rgb = cv2.cvtColor(v, cv2.COLOR_BGR2RGB)
            pose_res = self.pose_static.process(rgb)
            face_res = self.face_mesh_static.process(rgb)
            seg_res = self.selfie_seg_static.process(rgb)
            score = 0
            if pose_res and pose_res.pose_landmarks:
                score += 1
            if face_res and face_res.multi_face_landmarks:
                score += 1
            if score > best_score:
                best_score = score
                best = (pose_res, face_res, seg_res)
            if score == 2:
                break
        return best

    # ------------------------------------------------------------------
    # Full analysis pipeline
    # ------------------------------------------------------------------

    def analyze(self, frame_bgr: np.ndarray, image_mode: bool = False):
        """Run full pipeline on frozen frame. Updates self.result and self.matches."""
        h, w = frame_bgr.shape[:2]
        if image_mode:
            pose_res, face_res, seg_res = self._run_static_with_retries(frame_bgr)
        else:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pose_res = self.pose.process(rgb)
            face_res = self.face_mesh.process(rgb)
            seg_res = self.selfie_seg.process(rgb)

        seg_mask = seg_res.segmentation_mask if seg_res.segmentation_mask is not None else None

        # Calibrate normalizer
        face_lm = face_res.multi_face_landmarks[0] if face_res.multi_face_landmarks else None
        self.normalizer.calibrate(face_lm, w, h)

        if not self.normalizer.is_valid():
            # Still run but with degraded results
            print("Warning: Face not detected — body ratios may be approximate")

        pose_lm = pose_res.pose_landmarks if pose_res else None

        # Extract features
        body_vec = extract_body_vector(pose_lm, face_lm, self.normalizer, w, h)
        skin_data = extract_skin_tone(face_lm, frame_bgr, seg_mask)
        clothing_data = extract_clothing_colors(pose_lm, frame_bgr, seg_mask, self.normalizer)
        fashion_emb = None
        if self.enable_fashion_model:
            fashion_emb = extract_fashion_embedding(
                frame_bgr,
                pose_lm,
                self.normalizer,
                seg_mask,
                backend=self.embedding_backend,
            )
        self.last_body_vector = body_vec
        self.last_skin_data = skin_data
        self.last_clothing_data = clothing_data
        self.last_fashion_embedding = fashion_emb

        # Body-profile retrieval
        self.matches = []
        self.match_images = {}
        self.body_matches = []
        self.outfit_matches = []
        self.outfit_images = {}
        if body_vec:
            self.body_db.add_profile(body_vec, frame_bgr=frame_bgr)
            self.body_matches = self.body_db.search(
                body_vec,
                top_k=12,
                exclude_image_path=None,
                query_gender=self.query_gender,
                strict_gender=True,
            )
            print(f"Found {len(self.body_matches)} similar body profiles")
            self._print_body_match_list(self.body_matches)
            self._save_body_match_list(self.body_matches)

        # Outfit retrieval (dataset thumbnails)
        if fashion_emb is not None and self.db.is_ready():
            self.outfit_matches = self.db.search(fashion_emb, top_k=10)
            self.outfit_images = self.db.fetch_images_batch(self.outfit_matches[:10], size=(THUMB_W, THUMB_H))
            print(f"Found {len(self.outfit_matches)} similar outfits")
            self._save_outfit_match_assets(self.outfit_matches[:10], self.outfit_images)

        # Score (after retrieval so fashion match count can influence score)
        self.result = calculate_drip_score(
            body_vector=body_vec,
            skin_data=skin_data,
            clothing_data=clothing_data,
            fashion_embedding=fashion_emb,
            fashion_matches=self.outfit_matches,
            embedding_backend=self.embedding_backend,
            face_landmarks=face_lm,
            frame_bgr=frame_bgr,
            normalizer=self.normalizer,
        )

        # Primary display logic must be body-profile similarity.
        self.matches = self.body_matches
        self.match_images = {}

    def print_result_console(self):
        if not self.result:
            return
        print(f"\nDrip Score: {self.result['total']}/100  [{score_label(self.result['total'])}]")
        for k, v in self.result["breakdown"].items():
            print(f"  {k}: {v}")

    def run_single_image(self, image_path: str):
        """Analyze a single uploaded image and save a result panel."""
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        # Run core analysis pipeline
        self.analyze(img, image_mode=True)
        self.print_result_console()

        # Build and save a visual report panel
        panel = build_result_panel(
            img,
            self.result,
            self.matches,
            self.match_images,
            self.debug,
            self.match_scroll,
        )
        out_path = Path("cache") / f"result_{Path(image_path).stem}.jpg"
        out_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(out_path), panel)
        print(f"Saved result panel to: {out_path}")
        self._save_analysis_report(image_path=image_path)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _cleanup_mediapipe(self):
        self.pose.close()
        self.pose_static.close()
        self.face_mesh.close()
        self.face_mesh_static.close()
        self.selfie_seg.close()
        self.selfie_seg_static.close()

    def run_camera_capture(self, delay: int = 3):
        """Open camera, wait `delay` seconds, grab one frame, then analyse."""
        import time

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera")

        # Warm-up: cameras need a few frames for auto-exposure
        for _ in range(10):
            cap.read()

        print(f"Camera open — capturing in {delay}s … (strike a pose!)")
        time.sleep(delay)

        ret, frame = cap.read()
        cap.release()
        if not ret:
            raise RuntimeError("Failed to capture frame from camera")

        frame = cv2.flip(frame, 1)
        capture_path = Path("cache") / "camera_capture.jpg"
        capture_path.parent.mkdir(exist_ok=True)
        cv2.imwrite(str(capture_path), frame)
        print(f"Captured → {capture_path}")

        self.run_single_image(str(capture_path))

    def run(self, image_path: str | None = None, capture: bool = False):
        # ---- Load fashion model (optional) ----
        if self.enable_fashion_model:
            print(f"Fashion embedding: ENABLED  backend={self.embedding_backend}")
            if self.embedding_backend in {"hf_clip", "fashion_clip"} and os.getenv("DRIP_EMBED_SUBPROCESS", "1") == "1":
                print(f"{self.embedding_backend} load mode: subprocess isolation")
            else:
                load_fashion_model(backend=self.embedding_backend)
        else:
            print("Fashion embedding: DISABLED (set DRIP_ENABLE_FASHION_MODEL=1 to enable)")

        if not self.db.load():
            print("Outfit index not loaded — image inspirations disabled.")

        if capture:
            self.run_camera_capture()
            self._cleanup_mediapipe()
            return

        if image_path:
            self.run_single_image(image_path)
            self._cleanup_mediapipe()
            return

        # ---- Open preview window (webcam mode only) ----
        cv2.namedWindow("Drip Score", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Drip Score", WIN_W, WIN_H)

        # ---- Open webcam ----
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("\nDrip Score ready!  S=Scan  D=Debug  R=Reset  Q=Quit\n")

        while True:
            if self.state == "results" and self.result is not None:
                display = build_result_panel(
                    self.frozen_frame,
                    self.result,
                    self.matches,
                    self.match_images,
                    self.debug,
                    self.match_scroll,
                )
            elif self.state == "analyzing":
                display = loading_screen("Analyzing your drip...")
            else:
                # Live feed
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.flip(frame, 1)

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_res = self.pose.process(rgb)
                face_res = self.face_mesh.process(rgb)

                face_lm = (face_res.multi_face_landmarks[0]
                           if face_res.multi_face_landmarks else None)
                self.normalizer.calibrate(face_lm, w, h)

                display = draw_live_overlay(frame, pose_res, self.normalizer, self.debug)
                display = cv2.resize(display, (WIN_W, WIN_H))

            cv2.imshow("Drip Score", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == ord("s") and self.state == "live":
                ret, frame = cap.read()
                if ret:
                    self.frozen_frame = cv2.flip(frame, 1)
                    self.state = "analyzing"

                    # Show loading screen for 1 frame
                    cv2.imshow("Drip Score", loading_screen("Analyzing your drip..."))
                    cv2.waitKey(1)

                    try:
                        self.analyze(self.frozen_frame)
                        self.match_scroll = 0
                        self.state = "results"
                        self.print_result_console()
                    except Exception as e:
                        print(f"Analysis error: {e}")
                        import traceback
                        traceback.print_exc()
                        self.state = "live"

            elif key == ord("r"):
                self.state = "live"
                self.result = None
                self.matches = []
                self.match_images = {}

            elif key == ord("d"):
                self.debug = not self.debug

            elif ord("1") <= key <= ord("9") and self.state == "results":
                page = key - ord("1")
                self.match_scroll = page * RESULT_COLS
                # Lazy-load outfit thumbnails for paged results
                if self.matches and getattr(self.matches[0], "item_id", None) is not None and self.db.is_ready():
                    next_matches = self.matches[self.match_scroll: self.match_scroll + RESULT_COLS]
                    missing = [m for m in next_matches if m.item_id not in self.match_images]
                    if missing:
                        self.match_images.update(self.db.fetch_images_batch(missing, size=(THUMB_W, THUMB_H)))

        cap.release()
        cv2.destroyAllWindows()
        self._cleanup_mediapipe()


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drip Score analyzer")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Analyze a local image file instead of webcam feed",
    )
    parser.add_argument(
        "--capture",
        action="store_true",
        help="Grab a single frame from the camera (no GUI), then analyze it",
    )
    args = parser.parse_args()

    try:
        app = DripScoreApp()
        app.run(image_path=args.image, capture=args.capture)
        # One-shot modes occasionally leave native/worker resources alive
        # (HF/torch/multiprocessing). Force a clean terminal return.
        if args.image or args.capture:
            import sys
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)
    except RuntimeError as e:
        print(e)
        raise SystemExit(1)
