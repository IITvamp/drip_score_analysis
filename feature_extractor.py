"""
feature_extractor.py
--------------------
Extracts body vector, skin tone, clothing colors, and fashion embedding
from a single BGR frame using MediaPipe + Marqo-FashionSigLIP.
ALL pose measurements use PerspectiveNormalizer for camera-distance invariance.
"""

import math
import os
import sys
import subprocess
import tempfile
import cv2
import numpy as np
from sklearn.cluster import KMeans
import mediapipe as mp

from normalizer import PerspectiveNormalizer
from color_theory import rgb_to_lab, classify_skin_season


# ------------------------------------------------------------------
# MediaPipe landmark indices
# ------------------------------------------------------------------
LM = {
    "nose": 0,
    "l_ear": 7, "r_ear": 8,
    "l_shoulder": 11, "r_shoulder": 12,
    "l_elbow": 13, "r_elbow": 14,
    "l_wrist": 15, "r_wrist": 16,
    "l_hip": 23, "r_hip": 24,
    "l_knee": 25, "r_knee": 26,
    "l_ankle": 27, "r_ankle": 28,
}

FACE_FOREHEAD = [10, 151, 9, 8]
FACE_L_CHEEK = 234
FACE_R_CHEEK = 454


# ------------------------------------------------------------------
# Body shape classification
# ------------------------------------------------------------------

def _classify_body_shape(sh_ratio: float, hip_ratio: float, torso_leg: float) -> str:
    """
    sh_ratio  = shoulder_width / hip_width
    hip_ratio = hip_width / estimated_waist
    torso_leg = torso_height / (torso + leg height)
    """
    if sh_ratio > 1.18:
        return "inverted_triangle"
    elif sh_ratio < 0.88:
        return "triangle"
    elif 0.88 <= sh_ratio <= 1.12 and hip_ratio > 1.22:
        return "hourglass"
    elif torso_leg > 0.58:
        return "oval"
    else:
        return "rectangle"


# ------------------------------------------------------------------
# Body vector extraction
# ------------------------------------------------------------------

def extract_body_vector(pose_landmarks, face_landmarks, normalizer: PerspectiveNormalizer,
                        frame_w: int, frame_h: int) -> dict | None:
    """
    Returns a face-unit-normalised body vector dict, or None if confidence too low.
    """
    if pose_landmarks is None:
        return None

    lm = pose_landmarks.landmark

    # Visibility check — require key points visible
    required = [LM["l_shoulder"], LM["r_shoulder"], LM["l_hip"], LM["r_hip"]]
    if any(lm[i].visibility < 0.4 for i in required):
        return None

    def dist_px(i, j):
        dx = (lm[j].x - lm[i].x) * frame_w
        dy = (lm[j].y - lm[i].y) * frame_h
        return math.sqrt(dx * dx + dy * dy)

    use_face_units = normalizer.is_valid()

    # Fallback scale if face landmarks are unavailable.
    shoulder_w_px = max(1.0, dist_px(LM["l_shoulder"], LM["r_shoulder"]))
    scale_px = shoulder_w_px

    def dist_unit(i, j):
        if use_face_units:
            return normalizer.distance_face_units(lm[i], lm[j])
        return dist_px(i, j) / scale_px

    def point_dist_unit(p1, p2):
        if use_face_units:
            return normalizer.to_face_units(math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2))
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) / scale_px

    def midpoint_lm(i, j):
        """Return a pseudo-landmark at midpoint of two landmarks."""
        class _MidLM:
            x = (lm[i].x + lm[j].x) / 2
            y = (lm[i].y + lm[j].y) / 2
        return _MidLM()

    # Core measurements in face units
    shoulder_w = dist_unit(LM["l_shoulder"], LM["r_shoulder"])
    hip_w = dist_unit(LM["l_hip"], LM["r_hip"])
    mid_shoulder = midpoint_lm(LM["l_shoulder"], LM["r_shoulder"])
    mid_hip = midpoint_lm(LM["l_hip"], LM["r_hip"])
    torso_h = point_dist_unit(
        (mid_shoulder.x * frame_w, mid_shoulder.y * frame_h),
        (mid_hip.x * frame_w, mid_hip.y * frame_h),
    )

    ankle_visible = (lm[LM["l_ankle"]].visibility > 0.4 and
                     lm[LM["r_ankle"]].visibility > 0.4)
    if ankle_visible:
        mid_ankle = midpoint_lm(LM["l_ankle"], LM["r_ankle"])
        leg_h = point_dist_unit(
            (mid_hip.x * frame_w, mid_hip.y * frame_h),
            (mid_ankle.x * frame_w, mid_ankle.y * frame_h),
        )
    else:
        knee_visible = (lm[LM["l_knee"]].visibility > 0.4)
        if knee_visible:
            mid_knee = midpoint_lm(LM["l_knee"], LM["r_knee"])
            leg_h = point_dist_unit(
                (mid_hip.x * frame_w, mid_hip.y * frame_h),
                (mid_knee.x * frame_w, mid_knee.y * frame_h),
            ) * 1.8  # estimate
        else:
            leg_h = torso_h * 1.1  # fallback estimate

    # Derived ratios
    sh_ratio = shoulder_w / max(hip_w, 0.01)
    est_waist = hip_w * 0.82      # statistical proxy
    hip_waist_ratio = hip_w / max(est_waist, 0.01)
    total_h = torso_h + leg_h
    torso_leg_ratio = torso_h / max(total_h, 0.01)

    if use_face_units:
        # Height bucket: short < 5.6 fu, medium 5.6–7.0, tall > 7.0
        if total_h < 5.6:
            height_bucket = 0
        elif total_h < 7.0:
            height_bucket = 1
        else:
            height_bucket = 2
    else:
        # Approximate visible body fraction from shoulder-hip-ankle structure.
        body_fraction = min(1.0, max(0.0, (total_h * scale_px) / max(frame_h, 1)))
        if body_fraction < 0.45:
            height_bucket = 0
        elif body_fraction < 0.62:
            height_bucket = 1
        else:
            height_bucket = 2

    body_shape = _classify_body_shape(sh_ratio, hip_waist_ratio, torso_leg_ratio)

    # Average visibility as confidence
    conf = float(np.mean([lm[i].visibility for i in LM.values()
                          if lm[i].visibility > 0]))

    return {
        "shoulder_width_fu": round(shoulder_w, 3),
        "hip_width_fu": round(hip_w, 3),
        "torso_height_fu": round(torso_h, 3),
        "leg_height_fu": round(leg_h, 3),
        "shoulder_hip_ratio": round(sh_ratio, 3),
        "hip_waist_ratio": round(hip_waist_ratio, 3),
        "torso_leg_ratio": round(torso_leg_ratio, 3),
        "body_shape": body_shape,
        "height_bucket": height_bucket,
        "confidence": round(conf, 3),
    }


# ------------------------------------------------------------------
# Skin tone extraction
# ------------------------------------------------------------------

def extract_skin_tone(face_landmarks, frame_bgr: np.ndarray,
                      seg_mask: np.ndarray | None) -> dict | None:
    """
    Samples forehead and both cheeks, runs K-Means k=1, converts to LAB.
    Returns skin tone dict or None.
    """
    if face_landmarks is None:
        return None

    h, w = frame_bgr.shape[:2]
    lm = face_landmarks.landmark

    def px(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    # Sample patches
    samples = []
    patch_size = max(8, int(w * 0.012))

    sample_indices = FACE_FOREHEAD + [FACE_L_CHEEK, FACE_R_CHEEK]
    for idx in sample_indices:
        cx, cy = px(idx)
        x1 = max(0, cx - patch_size)
        x2 = min(w, cx + patch_size)
        y1 = max(0, cy - patch_size)
        y2 = min(h, cy + patch_size)
        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        # Apply seg mask if available
        if seg_mask is not None:
            mask_patch = seg_mask[y1:y2, x1:x2]
            flat = patch_rgb.reshape(-1, 3)
            flat_mask = mask_patch.reshape(-1) if mask_patch.ndim == 2 else mask_patch.reshape(-1)
            valid = flat[flat_mask > 0.5]
            if len(valid) > 10:
                samples.extend(valid.tolist())
                continue
        samples.extend(patch_rgb.reshape(-1, 3).tolist())

    if len(samples) < 20:
        return None

    pixels = np.array(samples, dtype=np.float32)
    # Remove very dark (shadow) and very light (glare) pixels
    mask = (pixels.mean(axis=1) > 40) & (pixels.mean(axis=1) < 230)
    pixels = pixels[mask]
    if len(pixels) < 10:
        return None

    km = KMeans(n_clusters=1, n_init=3, random_state=42)
    km.fit(pixels)
    dominant_rgb = km.cluster_centers_[0]
    r, g, b = int(dominant_rgb[0]), int(dominant_rgb[1]), int(dominant_rgb[2])

    L, a_val, b_val = rgb_to_lab(r, g, b)
    season = classify_skin_season(L, a_val, b_val)
    lightness_norm = L / 100.0
    warmth_norm = (a_val + 60) / 120.0  # map roughly -60..60 → 0..1

    return {
        "rgb": (r, g, b),
        "lab": (round(L, 2), round(a_val, 2), round(b_val, 2)),
        "skin_lightness": round(lightness_norm, 3),
        "skin_warmth": round(np.clip(warmth_norm, 0, 1), 3),
        "skin_season": season,
        "hex": f"#{r:02X}{g:02X}{b:02X}",
    }


# ------------------------------------------------------------------
# Clothing color extraction
# ------------------------------------------------------------------

def extract_clothing_colors(pose_landmarks, frame_bgr: np.ndarray,
                             seg_mask: np.ndarray | None,
                             normalizer: PerspectiveNormalizer) -> dict | None:
    """
    Crops torso and lower body regions, extracts dominant colors via K-Means.
    """
    if pose_landmarks is None:
        return None

    h, w = frame_bgr.shape[:2]
    lm = pose_landmarks.landmark

    def lm_px(idx):
        return (int(lm[idx].x * w), int(lm[idx].y * h))

    ls = lm_px(LM["l_shoulder"])
    rs = lm_px(LM["r_shoulder"])
    lh = lm_px(LM["l_hip"])
    rh = lm_px(LM["r_hip"])
    lk = lm_px(LM["l_knee"]) if lm[LM["l_knee"]].visibility > 0.3 else None
    rk = lm_px(LM["r_knee"]) if lm[LM["r_knee"]].visibility > 0.3 else None

    def crop_region(top_left, bot_right):
        x1 = max(0, min(top_left[0], bot_right[0]))
        x2 = min(w, max(top_left[0], bot_right[0]))
        y1 = max(0, min(top_left[1], bot_right[1]))
        y2 = min(h, max(top_left[1], bot_right[1]))
        return frame_bgr[y1:y2, x1:x2], seg_mask[y1:y2, x1:x2] if seg_mask is not None else None

    def dominant_colors(crop_bgr, crop_mask, k=3):
        if crop_bgr is None or crop_bgr.size == 0:
            return []
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pixels = crop_rgb.reshape(-1, 3).astype(np.float32)
        if crop_mask is not None:
            flat_mask = crop_mask.reshape(-1)
            if flat_mask.shape[0] == pixels.shape[0]:
                pixels = pixels[flat_mask > 0.5]
        if len(pixels) < 30:
            return []
        # Filter out near-black/white noise
        grey = pixels.mean(axis=1)
        pixels = pixels[(grey > 30) & (grey < 235)]
        if len(pixels) < 20:
            return []
        k_actual = min(k, len(pixels))
        km = KMeans(n_clusters=k_actual, n_init=3, random_state=42)
        km.fit(pixels)
        colors_rgb = [(int(c[0]), int(c[1]), int(c[2])) for c in km.cluster_centers_]
        return colors_rgb

    # Torso: shoulder-line to hip-line, padded inward 5%
    pad_x = int((rs[0] - ls[0]) * 0.05)
    torso_crop, torso_mask = crop_region(
        (ls[0] + pad_x, ls[1]),
        (rs[0] - pad_x, rh[1])
    )
    torso_colors = dominant_colors(torso_crop, torso_mask)

    # Lower body: hip-line to knee-line (or bottom of frame)
    if lk and rk:
        lower_crop, lower_mask = crop_region((lh[0], lh[1]), (rk[0], rk[1]))
    else:
        lower_crop, lower_mask = crop_region((lh[0], lh[1]), (rh[0], min(h, rh[1] + 200)))
    lower_colors = dominant_colors(lower_crop, lower_mask)

    # Convert to LAB
    torso_lab = [rgb_to_lab(*c) for c in torso_colors]
    lower_lab = [rgb_to_lab(*c) for c in lower_colors]

    # Count distinct colors across full outfit
    all_colors = torso_colors + lower_colors
    distinct = _count_distinct_colors(all_colors)

    return {
        "torso_colors_rgb": torso_colors,
        "lower_colors_rgb": lower_colors,
        "torso_dominant_lab": torso_lab[0] if torso_lab else None,
        "lower_dominant_lab": lower_lab[0] if lower_lab else None,
        "torso_colors_lab": torso_lab,
        "lower_colors_lab": lower_lab,
        "num_distinct_colors": distinct,
    }


def _count_distinct_colors(colors_rgb: list, threshold: float = 40.0) -> int:
    """Count perceptually distinct colors (LAB delta-E > threshold)."""
    if not colors_rgb:
        return 0
    labs = [rgb_to_lab(*c) for c in colors_rgb]
    distinct = 1
    for i in range(1, len(labs)):
        L1, a1, b1 = labs[i]
        is_new = True
        for j in range(i):
            L2, a2, b2 = labs[j]
            delta_e = math.sqrt((L1-L2)**2 + (a1-a2)**2 + (b1-b2)**2)
            if delta_e < threshold:
                is_new = False
                break
        if is_new:
            distinct += 1
    return distinct


# ------------------------------------------------------------------
# Fashion embedding via Marqo-FashionSigLIP
# ------------------------------------------------------------------

_fashion_model = None
_fashion_preprocess = None
_fashion_tokenizer = None
_fashion_processor = None
_fashion_backend = None

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


def load_fashion_model(backend: str | None = None):
    """
    Load and cache the selected fashion embedding model.
    Supported backends: 'hf_clip' (default), 'marqo_siglip'.
    """
    global _fashion_model, _fashion_preprocess, _fashion_tokenizer, _fashion_processor, _fashion_backend

    selected = _resolve_backend(backend)

    if _fashion_model is not None and _fashion_backend == selected:
        return _fashion_model, _fashion_preprocess, _fashion_processor

    if selected == "marqo_siglip":
        import open_clip

        print("Loading Marqo-FashionSigLIP model (first run may download ~600MB)...")
        model, _, preprocess_val = open_clip.create_model_and_transforms(
            "hf-hub:Marqo/marqo-fashionSigLIP"
        )
        model.eval()
        _fashion_model = model
        _fashion_preprocess = preprocess_val
        _fashion_tokenizer = open_clip.get_tokenizer("hf-hub:Marqo/marqo-fashionSigLIP")
        _fashion_processor = None
        _fashion_backend = selected
        print("Model loaded.")
        return _fashion_model, _fashion_preprocess, _fashion_processor

    # Stable default backend
    from transformers import CLIPModel, CLIPProcessor

    model_id = _backend_model_id(selected)
    print(f"Loading HF CLIP model ({model_id})...")
    processor = CLIPProcessor.from_pretrained(model_id, use_fast=False)
    model = CLIPModel.from_pretrained(model_id)
    model.eval()
    _fashion_model = model
    _fashion_preprocess = None
    _fashion_tokenizer = None
    _fashion_processor = processor
    _fashion_backend = selected
    print("Model loaded.")
    return _fashion_model, _fashion_preprocess, _fashion_processor


def extract_fashion_embedding(frame_bgr: np.ndarray, pose_landmarks,
                               normalizer: PerspectiveNormalizer,
                               seg_mask: np.ndarray | None,
                               backend: str | None = None) -> np.ndarray | None:
    """
    Crop the full outfit (shoulders to ankles / bottom of frame),
    apply segmentation mask, embed with FashionSigLIP.
    Returns L2-normalised 768-dim numpy array or None.
    """
    from PIL import Image

    selected = _resolve_backend(backend)

    h, w = frame_bgr.shape[:2]

    if pose_landmarks is not None and normalizer.is_valid():
        lm = pose_landmarks.landmark
        ls = (int(lm[LM["l_shoulder"]].x * w), int(lm[LM["l_shoulder"]].y * h))
        rs = (int(lm[LM["r_shoulder"]].x * w), int(lm[LM["r_shoulder"]].y * h))

        if lm[LM["l_ankle"]].visibility > 0.3:
            la = (int(lm[LM["l_ankle"]].x * w), int(lm[LM["l_ankle"]].y * h))
            ra = (int(lm[LM["r_ankle"]].x * w), int(lm[LM["r_ankle"]].y * h))
            y2 = max(la[1], ra[1]) + 20
        else:
            y2 = h

        x1 = max(0, min(ls[0], rs[0]) - 20)
        x2 = min(w, max(ls[0], rs[0]) + 20)
        y1 = max(0, min(ls[1], rs[1]) - 10)
        y2 = min(h, y2)
    else:
        x1, y1, x2, y2 = 0, 0, w, h

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        crop = frame_bgr

    # Apply segmentation mask (white background behind person)
    if seg_mask is not None:
        mask_crop = seg_mask[y1:y2, x1:x2]
        if mask_crop.shape[:2] == crop.shape[:2]:
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            bg = np.full_like(crop_rgb, 255)
            if mask_crop.ndim == 2:
                alpha = (mask_crop[..., None] > 0.3).astype(np.float32)
            else:
                alpha = (mask_crop > 0.3).astype(np.float32)
            crop_rgb = (crop_rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            img = Image.fromarray(crop_rgb)
        else:
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    else:
        img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

    if selected in {"hf_clip", "fashion_clip"} and os.getenv("DRIP_EMBED_SUBPROCESS", "1") == "1":
        # Isolate HF model load in subprocess to avoid native crashes when
        # MediaPipe/OpenCV are already loaded in the parent process.
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_img:
            img_path = tmp_img.name
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_out:
            out_path = tmp_out.name
        try:
            img.save(img_path)
            model_id = _backend_model_id(selected)
            worker_python = sys.executable
            if selected == "fashion_clip":
                default_fc_python = os.path.abspath(".venv-fashionclip/bin/python")
                worker_python = os.getenv("DRIP_FASHIONCLIP_PYTHON", default_fc_python)

            child_code = """
import os
import sys
import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

model_id = sys.argv[3]
proc = CLIPProcessor.from_pretrained(model_id, use_fast=False)
model = CLIPModel.from_pretrained(model_id)
model.eval()

im = Image.open(sys.argv[1]).convert("RGB")
with torch.no_grad():
    inp = proc(images=im, return_tensors="pt")
    f = model.get_image_features(**inp)
    if not torch.is_tensor(f):
        if hasattr(f, "pooler_output") and f.pooler_output is not None:
            f = f.pooler_output
        elif hasattr(f, "last_hidden_state"):
            f = f.last_hidden_state.mean(dim=1)
        else:
            raise RuntimeError("Unsupported CLIP output type from get_image_features")
    f = f / f.norm(dim=-1, keepdim=True)

np.save(sys.argv[2], f.squeeze(0).cpu().numpy().astype(np.float32))
"""
            # Subprocess can occasionally hang (model download/metal init/etc).
            # Use a timeout so camera capture never blocks forever.
            timeout_s = float(os.getenv("DRIP_EMBED_SUBPROCESS_TIMEOUT_S", "120"))
            proc = subprocess.run(
                [worker_python, "-c", child_code, img_path, out_path, model_id],
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_s,
            )
            if proc.returncode != 0:
                # For FashionCLIP, fallback to dedicated worker script path.
                if selected == "fashion_clip":
                    worker_script = os.path.abspath("fashion_clip_worker.py")
                    proc2 = subprocess.run(
                        [
                            worker_python,
                            worker_script,
                            "--image",
                            img_path,
                            "--output",
                            out_path,
                            "--model-id",
                            model_id,
                        ],
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    if proc2.returncode != 0:
                        print(
                            f"FashionCLIP worker failed (returncode={proc2.returncode}): "
                            f"{proc2.stderr.strip()}"
                        )
                        return None
                else:
                    print(
                        f"CLIP subprocess failed (backend={selected}, returncode={proc.returncode}): "
                        f"{proc.stderr.strip()}"
                    )
                    return None
            vec = np.load(out_path)
            return vec.astype(np.float32)
        except subprocess.TimeoutExpired:
            print(
                f"CLIP subprocess timed out after {os.getenv('DRIP_EMBED_SUBPROCESS_TIMEOUT_S','120')}s "
                f"(backend={selected}). Skipping fashion embedding for this run."
            )
            return None
        finally:
            for p in (img_path, out_path):
                try:
                    os.remove(p)
                except OSError:
                    pass

    try:
        model, preprocess, processor = load_fashion_model(backend=selected)
        import torch
        with torch.no_grad():
            if selected == "marqo_siglip":
                tensor = preprocess(img).unsqueeze(0)
                features = model.encode_image(tensor)
            else:
                inputs = processor(images=img, return_tensors="pt")
                features = model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy().astype(np.float32)
    except Exception as e:
        print(f"Fashion embedding unavailable (backend={selected}): {e}")
        return None
