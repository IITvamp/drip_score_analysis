"""
scorer.py
---------
Calculates the Drip Score out of 100 across six pillars.
All inputs come from feature_extractor outputs plus fashion retrieval matches.
"""

import numpy as np
from color_theory import score_color_season_match, score_color_harmony

from vector_db import FashionMatch


# ------------------------------------------------------------------
# Scoring config (easy to tweak in one place)
# ------------------------------------------------------------------

FIT_MAX = 20
HARMONY_MAX = 20
SEASON_MAX = 15
GROOMING_MAX = 15
COHERENCE_MAX = 10
FASHION_RETRIEVAL_MAX = 20

# Fashion retrieval scoring knobs
FASHION_SIMILARITY_THRESHOLD = 0.80
FASHION_POINTS_PER_MATCH = 2


# ------------------------------------------------------------------
# Body shape fit rules
# ------------------------------------------------------------------

FIT_RULES = {
    "inverted_triangle": {
        "ideal_text_cues": [
            "slim fit", "tapered trousers", "fitted shirt", "slim chinos",
            "skinny jeans", "fitted", "tailored"
        ],
        "bad_text_cues": [
            "oversized jacket", "baggy", "wide-leg trousers", "loose fit"
        ],
        "style_tip": "Slim-cut bottoms balance your broad shoulders. Avoid adding volume on top.",
    },
    "triangle": {
        "ideal_text_cues": [
            "structured jacket", "blazer", "patterned shirt", "detailed top",
            "wide shoulders", "fitted jacket"
        ],
        "bad_text_cues": [
            "wide-leg trousers", "flare", "light colored wide pants", "baggy bottom"
        ],
        "style_tip": "Add structure and detail on top to balance wider hips.",
    },
    "rectangle": {
        "ideal_text_cues": [
            "layered", "jacket", "blazer", "structured", "belted",
            "fitted", "tailored suit"
        ],
        "bad_text_cues": [
            "shapeless", "oversized", "baggy", "loose fit all over"
        ],
        "style_tip": "Layers and structure create the illusion of shape. Belted waists help.",
    },
    "hourglass": {
        "ideal_text_cues": [
            "fitted", "tailored", "slim fit", "wrap", "form-fitting"
        ],
        "bad_text_cues": [
            "oversized", "extremely baggy", "shapeless"
        ],
        "style_tip": "Fitted clothes highlight your natural proportions. Avoid extreme oversize.",
    },
    "oval": {
        "ideal_text_cues": [
            "v-neck", "vertical stripes", "monochromatic", "structured jacket",
            "dark colors", "fitted jacket", "vertical"
        ],
        "bad_text_cues": [
            "horizontal stripes", "tight all over", "tucked-in", "crop top"
        ],
        "style_tip": "Vertical lines and monochromatic looks elongate your silhouette.",
    },
}


# ------------------------------------------------------------------
# Fit & proportion scoring
# ------------------------------------------------------------------

def score_fit_proportion(body_vector: dict, clothing_data: dict | None,
                          fashion_embedding: np.ndarray | None) -> tuple[int, str]:
    """
    Score 0-FIT_MAX: Does the outfit work for this body shape?
    Uses outfit description text from vector DB matches if available.
    Falls back to clothing color analysis.
    """
    if not body_vector:
        return int(FIT_MAX * 0.5), "Could not detect body shape"

    body_shape = body_vector.get("body_shape", "rectangle")
    confidence = body_vector.get("confidence", 0.5)

    rules = FIT_RULES.get(body_shape, FIT_RULES["rectangle"])
    tip = rules["style_tip"]

    # Without a garment-description oracle, give a moderate base
    # and adjust based on silhouette proportions
    torso_leg = body_vector.get("torso_leg_ratio", 0.5)
    sh_ratio = body_vector.get("shoulder_hip_ratio", 1.0)

    # Reward proportions that suggest well-fitted clothing
    # (Very extreme ratios with high confidence usually mean fitted clothes)
    base = 15
    if confidence > 0.7:
        base = 18
    if confidence > 0.85:
        base = 20

    score_25_scale = min(25, int(base * (0.8 + 0.4 * confidence)))
    score = int(round((score_25_scale / 25.0) * FIT_MAX))
    return score, tip


# ------------------------------------------------------------------
# Grooming scoring
# ------------------------------------------------------------------

def score_grooming(face_landmarks, frame_bgr, normalizer) -> tuple[int, str]:
    """
    Score 0-GROOMING_MAX based on detectable grooming signals.
    Measures hair-edge variance, collar/accessory presence.
    """
    if face_landmarks is None or not normalizer.is_valid():
        return int(GROOMING_MAX * 0.53), "Face not detected clearly — grooming score estimated"

    import cv2
    import numpy as np

    h, w = frame_bgr.shape[:2]
    lm = face_landmarks.landmark

    # Hair region: above the forehead landmark
    fh_y = int(lm[10].y * h)
    fh_x = int(lm[10].x * w)
    fu = normalizer.face_unit_px

    # Crop hair strip
    hair_y1 = max(0, fh_y - int(fu * 1.2))
    hair_y2 = max(0, fh_y - int(fu * 0.1))
    hair_x1 = max(0, fh_x - int(fu * 1.5))
    hair_x2 = min(w, fh_x + int(fu * 1.5))

    hair_score = 5  # default
    if hair_y2 > hair_y1 and hair_x2 > hair_x1:
        hair_crop = frame_bgr[hair_y1:hair_y2, hair_x1:hair_x2]
        if hair_crop.size > 0:
            gray = cv2.cvtColor(hair_crop, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 30, 100)
            edge_density = edges.mean()
            # Lower edge density in hair region = neater hair
            if edge_density < 8:
                hair_score = 7   # very neat
            elif edge_density < 18:
                hair_score = 5   # average
            else:
                hair_score = 3   # messy / flyaways

    # Collar region: below chin landmark (5)
    chin_y = int(lm[152].y * h)
    chin_x = int(lm[152].x * w)
    collar_y1 = chin_y
    collar_y2 = min(h, chin_y + int(fu * 1.5))
    collar_x1 = max(0, chin_x - int(fu * 2.0))
    collar_x2 = min(w, chin_x + int(fu * 2.0))

    collar_score = 4  # default
    if collar_y2 > collar_y1 and collar_x2 > collar_x1:
        collar_crop = frame_bgr[collar_y1:collar_y2, collar_x1:collar_x2]
        if collar_crop.size > 0:
            gray = cv2.cvtColor(collar_crop, cv2.COLOR_BGR2GRAY)
            # High contrast near collar = structured collar (shirt/jacket)
            contrast = gray.std()
            if contrast > 35:
                collar_score = 5   # structured collar
            elif contrast > 18:
                collar_score = 4   # some collar presence
            else:
                collar_score = 2   # open/no collar

    # Wrist accessory proxy: check wrist landmark visibility
    watch_bonus = 0
    if lm[15].visibility > 0.6 or lm[16].visibility > 0.6:
        watch_bonus = 3

    score = min(GROOMING_MAX, hair_score + collar_score + watch_bonus)
    msgs = []
    if hair_score >= 6:
        msgs.append("Hair looks neat")
    else:
        msgs.append("Tidy up the hair")
    if collar_score >= 4:
        msgs.append("good collar presence")
    if watch_bonus:
        msgs.append("accessory detected")

    return score, " · ".join(msgs) if msgs else "Average grooming"


# ------------------------------------------------------------------
# Style coherence scoring
# ------------------------------------------------------------------

def score_style_coherence(fashion_embedding: np.ndarray | None,
                           clothing_data: dict | None) -> tuple[int, str]:
    """
    Score 0-COHERENCE_MAX: Does the outfit tell one consistent story?
    Uses embedding distance between torso and lower-body crops when available.
    Falls back to color distinctness heuristic.
    """
    if fashion_embedding is None:
        if clothing_data:
            distinct = clothing_data.get("num_distinct_colors", 2)
            if distinct <= 2:
                return min(COHERENCE_MAX, 9), "Monochromatic or minimal palette — coherent"
            elif distinct == 3:
                return min(COHERENCE_MAX, 7), "Three distinct colors — manageable"
            else:
                return min(COHERENCE_MAX, 4), f"Too many colors ({distinct}) — pick a palette"
        return min(COHERENCE_MAX, 5), "Could not evaluate style coherence"

    # The embedding encodes the full outfit — high norm means distinctive style signal
    norm = float(np.linalg.norm(fashion_embedding))
    if norm > 0.95:
        score = 9
        msg = "Strong, coherent style signal"
    elif norm > 0.75:
        score = 7
        msg = "Decent style coherence"
    else:
        score = 4
        msg = "Mixed signals — the outfit has competing styles"

    if clothing_data:
        distinct = clothing_data.get("num_distinct_colors", 2)
        if distinct > 4:
            score = max(2, score - 3)
            msg += f" (−pts: {distinct} colors is too busy)"

    return min(COHERENCE_MAX, score), msg


def score_fashion_retrieval(
    fashion_matches: list[FashionMatch] | None,
    embedding_backend: str = "hf_clip",
) -> tuple[int, str]:
    """
    Score 0-FASHION_RETRIEVAL_MAX based on count of similar fashion images.

    Rule:
    - Count matches with similarity >= FASHION_SIMILARITY_THRESHOLD
    - Give FASHION_POINTS_PER_MATCH points each
    - Cap at FASHION_RETRIEVAL_MAX
    """
    backend = (embedding_backend or "").strip().lower()
    if backend != "hf_clip":
        return 0, "Fashion retrieval score active only with hf_clip backend"

    if not fashion_matches:
        return 0, "No fashion matches found"

    strong_count = sum(
        1
        for m in fashion_matches
        if float(getattr(m, "similarity_score", 0.0)) >= FASHION_SIMILARITY_THRESHOLD
    )
    score = min(FASHION_RETRIEVAL_MAX, strong_count * FASHION_POINTS_PER_MATCH)
    return score, (
        f"{strong_count} matches >= {FASHION_SIMILARITY_THRESHOLD:.2f}; "
        f"{FASHION_POINTS_PER_MATCH} pts each"
    )


def compute_outfit_match_delta(fashion_matches: list[FashionMatch] | None) -> tuple[dict, str]:
    """
    Diagnostic factor (non-scoring): quantify separation between top matches.
    Higher spread means the best match is distinctly better than nearby neighbors.
    """
    if not fashion_matches:
        diag = {
            "top1": 0.0,
            "top2": 0.0,
            "top5_avg": 0.0,
            "top1_top2_gap": 0.0,
            "top1_top5_gap": 0.0,
            "spread_score": 0.0,
        }
        return diag, "No outfit matches for delta analysis"

    scores = [float(getattr(m, "similarity_score", 0.0)) for m in fashion_matches]
    scores = [float(np.clip(s, 0.0, 1.0)) for s in scores]
    top1 = scores[0] if len(scores) >= 1 else 0.0
    top2 = scores[1] if len(scores) >= 2 else top1
    top5_avg = float(np.mean(scores[: min(5, len(scores))])) if scores else 0.0

    top1_top2_gap = max(0.0, top1 - top2)
    top1_top5_gap = max(0.0, top1 - top5_avg)
    spread_score = float(np.clip(top1_top5_gap / 0.20, 0.0, 1.0))

    diag = {
        "top1": round(top1, 4),
        "top2": round(top2, 4),
        "top5_avg": round(top5_avg, 4),
        "top1_top2_gap": round(top1_top2_gap, 4),
        "top1_top5_gap": round(top1_top5_gap, 4),
        "spread_score": round(spread_score, 4),
    }
    tip = (
        f"Top1={diag['top1']:.3f}, Top2={diag['top2']:.3f}, "
        f"gap(1-5avg)={diag['top1_top5_gap']:.3f}"
    )
    return diag, tip


def build_why_this_score_summary(
    total: int,
    core_score: int,
    core_max: int,
    fit_n: float,
    harmony_n: float,
    grooming_n: float,
    coherence_n: float,
    maha_sq: float,
    base_norm: float,
    fit_gate: float,
    final_norm: float,
    fashion_score: int,
    fashion_tip: str,
    hard_veto_triggered: bool,
) -> str:
    """
    Build a compact human-readable explanation for why this total happened.
    """
    pillar_vals = {
        "fit": fit_n,
        "color harmony": harmony_n,
        "grooming": grooming_n,
        "coherence": coherence_n,
    }
    weakest = sorted(pillar_vals.items(), key=lambda kv: kv[1])[:2]
    strongest = sorted(pillar_vals.items(), key=lambda kv: kv[1], reverse=True)[:2]

    weak_text = ", ".join(f"{k} ({v:.2f})" for k, v in weakest)
    strong_text = ", ".join(f"{k} ({v:.2f})" for k, v in strongest)

    veto_text = (
        "Hard fit veto triggered (fit < 0.30), so score was capped at 25."
        if hard_veto_triggered
        else ""
    )

    return (
        f"Final score is {total}/100 (core {core_score}/{core_max} + fashion {fashion_score}/20). "
        f"Mahalanobis distance (d²={maha_sq:.3f}) gives base confidence {base_norm:.3f}, "
        f"fit-gated (fit^1.2={fit_gate:.3f}) to {final_norm:.3f} → {core_score}/{core_max} core pts. "
        f"Strongest pillars: {strong_text}. Weakest pillars: {weak_text}. "
        f"{veto_text} Fashion retrieval: {fashion_score}/20 ({fashion_tip})."
    ).strip()


def build_improvement_suggestions(
    total: int,
    fit_n: float,
    harmony_n: float,
    grooming_n: float,
    coherence_n: float,
    fashion_score: int,
    body_shape: str,
    fit_tip: str,
    harmony_tip: str,
    groom_tip: str,
    coherence_tip: str,
    fashion_tip: str,
    num_fashion_matches: int,
) -> list[str]:
    """
    Generate ranked, actionable natural-language suggestions for what
    the person could improve to raise their score.
    Only surfaces items that are actually weak — strong pillars are skipped.
    """
    suggestions: list[str] = []

    SHAPE_ADVICE = {
        "inverted_triangle": (
            "Your shoulders are wider than your hips. "
            "Slim-fit or tapered trousers and fitted shirts work well. "
            "Avoid oversized tops or shoulder pads — they'll over-emphasize your upper body."
        ),
        "triangle": (
            "Your hips are wider than your shoulders. "
            "Structured jackets, blazers, and detailed tops will add visual width on top. "
            "Avoid bright or patterned bottoms that draw the eye downward."
        ),
        "rectangle": (
            "Your shoulders and hips are roughly even. "
            "Layering, belted waists, and structured outerwear will create shape. "
            "A well-fitted blazer instantly adds definition."
        ),
        "hourglass": (
            "You have balanced proportions — fitted, form-following clothes showcase that. "
            "Wrap styles and tailored cuts are your best friend. "
            "Avoid extremely baggy outfits that hide your natural shape."
        ),
        "oval": (
            "V-necks, vertical stripes, and monochromatic outfits elongate your silhouette. "
            "Dark, structured jackets create a clean vertical line. "
            "Avoid horizontal stripes and tucked-in shirts."
        ),
    }

    pillar_scores = [
        ("fit", fit_n, fit_tip),
        ("color_harmony", harmony_n, harmony_tip),
        ("grooming", grooming_n, groom_tip),
        ("coherence", coherence_n, coherence_tip),
    ]
    weak = [(name, val, tip) for name, val, tip in pillar_scores if val < 0.75]
    weak.sort(key=lambda t: t[1])

    for name, val, tip in weak:
        pct = int(val * 100)
        if name == "fit":
            shape_help = SHAPE_ADVICE.get(body_shape, "")
            suggestions.append(
                f"Fit & Proportion ({pct}%): Your outfit-to-body-shape match could be stronger. "
                f"{shape_help}"
            )
        elif name == "color_harmony":
            if "neutral" in tip.lower() or "monochrom" in tip.lower():
                suggestions.append(
                    f"Color Harmony ({pct}%): Your palette is very neutral/safe. "
                    f"Try introducing one bold accent color (a watch, pocket square, shoes, or bag) "
                    f"to add visual interest without clashing."
                )
            elif "too many" in tip.lower() or "busy" in tip.lower():
                suggestions.append(
                    f"Color Harmony ({pct}%): Too many competing colors. "
                    f"Stick to 2-3 colors max — one dominant, one secondary, one accent. "
                    f"Neutral base + one pop of color is a reliable formula."
                )
            else:
                suggestions.append(
                    f"Color Harmony ({pct}%): The color coordination between your top and bottom "
                    f"could be tighter. Try pairing complementary tones or using a monochrome base."
                )
        elif name == "grooming":
            if "not detected" in tip.lower() or "estimated" in tip.lower():
                suggestions.append(
                    f"Grooming ({pct}%): We couldn't clearly detect your face — "
                    f"for a better score, use a well-lit photo where your face, hair, and "
                    f"collar/neckline are clearly visible."
                )
            else:
                parts = []
                if "tidy" in tip.lower() or "hair" in tip.lower():
                    parts.append("neaten up your hairstyle")
                if "collar" not in tip.lower() or "no collar" in tip.lower():
                    parts.append("add a structured collar (shirt, polo, or jacket)")
                if "accessory" not in tip.lower():
                    parts.append("consider a visible accessory like a watch or bracelet")
                if parts:
                    suggestions.append(
                        f"Grooming ({pct}%): {'; '.join(parts).capitalize()}."
                    )
                else:
                    suggestions.append(
                        f"Grooming ({pct}%): Small details matter — tidy hair, a clean collar, "
                        f"and one accessory can elevate your look significantly."
                    )
        elif name == "coherence":
            suggestions.append(
                f"Style Coherence ({pct}%): Your outfit is sending mixed style signals. "
                f"Pick one style direction (e.g. smart-casual, streetwear, minimal) and "
                f"make sure every piece — shoes included — belongs to that story."
            )

    if fashion_score < FASHION_RETRIEVAL_MAX * 0.5:
        if num_fashion_matches == 0:
            suggestions.append(
                "Fashion Match: No similar outfits were found in our database. "
                "This could mean your look is very unique, or the photo didn't capture "
                "enough clothing detail. Try a full-body shot with clear lighting."
            )
        else:
            suggestions.append(
                f"Fashion Match ({fashion_score}/{FASHION_RETRIEVAL_MAX}): Only a few strong "
                f"matches were found. Outfits that align with well-established fashion categories "
                f"(smart-casual, athleisure, tailored formal) tend to score higher here."
            )

    if not suggestions:
        if total >= 85:
            suggestions.append(
                "You're looking great! To push even higher, focus on fine details: "
                "a statement accessory, perfectly matching belt-to-shoe color, "
                "or a pocket square that picks up your shirt's secondary tone."
            )
        else:
            suggestions.append(
                "No single area is very weak, but small improvements across the board "
                "will add up. Focus on tightening color coordination and ensuring your "
                "outfit silhouette matches your body shape."
            )

    return suggestions


# ------------------------------------------------------------------
# Master scoring function
# ------------------------------------------------------------------

def calculate_drip_score(
    body_vector: dict | None,
    skin_data: dict | None,
    clothing_data: dict | None,
    fashion_embedding: np.ndarray | None,
    fashion_matches: list[FashionMatch] | None = None,
    embedding_backend: str = "hf_clip",
    face_landmarks=None,
    frame_bgr=None,
    normalizer=None,
) -> dict:
    """
    Calculate full Drip Score (0-100) across 6 pillars.
    Returns complete breakdown dict.
    """

    # ---- Pillar 1: Fit & Proportion (20 pts) ----
    fit_score, fit_tip = score_fit_proportion(
        body_vector, clothing_data, fashion_embedding
    )

    # ---- Pillar 2: Color Harmony (20 pts) ----
    if clothing_data and clothing_data.get("torso_colors_lab") and clothing_data.get("lower_colors_lab"):
        raw_harmony_score, harmony_tip = score_color_harmony(
            clothing_data["torso_colors_lab"],
            clothing_data["lower_colors_lab"],
            clothing_data.get("num_distinct_colors", 2),
        )
        harmony_score = int(round((raw_harmony_score / 25.0) * HARMONY_MAX))
    else:
        harmony_score, harmony_tip = int(HARMONY_MAX * 0.48), "Could not analyze clothing colors"

    # ---- Pillar 3: Color Season Match (15 pts, informational only) ----
    # Disabled from core total logic to avoid background/color-cast instability.
    season_score, season_tip = int(SEASON_MAX * 0.5), (
        "Color season factor disabled from total score (informational only)"
    )

    # ---- Pillar 4: Grooming (15 pts) ----
    if face_landmarks is not None and frame_bgr is not None and normalizer is not None:
        groom_score, groom_tip = score_grooming(face_landmarks, frame_bgr, normalizer)
    else:
        groom_score, groom_tip = int(GROOMING_MAX * 0.53), "Face not detected — grooming estimated"

    # ---- Pillar 5: Style Coherence (10 pts) ----
    coher_score, coher_tip = score_style_coherence(fashion_embedding, clothing_data)

    # ---- Pillar 6: Fashion retrieval quality (20 pts) ----
    fashion_score, fashion_tip = score_fashion_retrieval(
        fashion_matches=fashion_matches,
        embedding_backend=embedding_backend,
    )
    outfit_delta_diag, outfit_delta_tip = compute_outfit_match_delta(fashion_matches)

    # Convert core pillars to 0..1 so hybrid math stays scale-invariant.
    fit_n = float(np.clip(fit_score / max(FIT_MAX, 1), 0.0, 1.0))
    harmony_n = float(np.clip(harmony_score / max(HARMONY_MAX, 1), 0.0, 1.0))
    grooming_n = float(np.clip(groom_score / max(GROOMING_MAX, 1), 0.0, 1.0))
    coherence_n = float(np.clip(coher_score / max(COHERENCE_MAX, 1), 0.0, 1.0))

    # Core Mahalanobis scores 0-CORE_MAX_POINTS; fashion adds on top for 0-100.
    CORE_MAX_POINTS = 100 - FASHION_RETRIEVAL_MAX          # 80

    # Ideal = perfect score on every pillar; zero distance = full core points.
    x = np.array([fit_n, harmony_n, grooming_n, coherence_n], dtype=np.float32)
    ideal = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    # Wide covariance so moderate deviations (0.2-0.3) are tolerable.
    # Off-diagonals encode expected correlations (grooming <-> coherence).
    cov = np.array(
        [
            [0.40, 0.06, 0.03, 0.03],
            [0.06, 0.40, 0.04, 0.04],
            [0.03, 0.04, 0.40, 0.15],
            [0.03, 0.04, 0.15, 0.40],
        ],
        dtype=np.float32,
    )
    cov_inv = np.linalg.pinv(cov)
    diff = x - ideal
    maha_sq = float(diff.T @ cov_inv @ diff)
    base_norm = float(np.exp(-0.5 * maha_sq))

    # Fit gate: multiplicative veto so poor fit suppresses other pillars.
    FIT_GATE_EXP = 1.2
    fit_gate = float(np.clip(fit_n, 0.0, 1.0) ** FIT_GATE_EXP)
    final_norm = float(np.clip(base_norm * fit_gate, 0.0, 1.0))
    core_score = int(round(final_norm * CORE_MAX_POINTS))

    # Total = core (0-80) + fashion retrieval (0-20) = 0-100
    total = core_score + fashion_score

    # Hard veto floor: if fit is abysmal, cap total regardless of other pillars.
    if fit_n < 0.30:
        total = min(total, 25)

    total = min(100, max(0, total))

    score_calculation = {
        "normalized_inputs": {
            "fit": round(fit_n, 6),
            "color_harmony": round(harmony_n, 6),
            "grooming": round(grooming_n, 6),
            "coherence": round(coherence_n, 6),
            "color_season": "disabled_from_total",
        },
        "mahalanobis": {
            "vector_x": [round(float(v), 6) for v in x.tolist()],
            "ideal_vector": [round(float(v), 6) for v in ideal.tolist()],
            "covariance_matrix": [[round(float(v), 6) for v in row] for row in cov.tolist()],
            "distance_squared": round(maha_sq, 6),
            "base_norm": round(base_norm, 6),
            "base_score_0_core": round(base_norm * CORE_MAX_POINTS, 3),
        },
        "fit_gate": {
            "fit_value": round(fit_n, 6),
            "exponent": FIT_GATE_EXP,
            "gate_value": round(fit_gate, 6),
            "gated_norm": round(final_norm, 6),
            "core_score": int(core_score),
            "core_max": int(CORE_MAX_POINTS),
        },
        "hard_veto": {
            "threshold": 0.3,
            "cap_when_below_threshold": 25,
            "triggered": bool(fit_n < 0.30),
        },
        "fashion_component": {
            "score_points": int(fashion_score),
            "max_points": int(FASHION_RETRIEVAL_MAX),
        },
        "final_total": int(total),
        "formula": "total = round(exp(-0.5 * d²) * fit^1.2 * 80) + fashion_score",
    }
    why_this_score = build_why_this_score_summary(
        total=total,
        core_score=core_score,
        core_max=CORE_MAX_POINTS,
        fit_n=fit_n,
        harmony_n=harmony_n,
        grooming_n=grooming_n,
        coherence_n=coherence_n,
        maha_sq=maha_sq,
        base_norm=base_norm,
        fit_gate=fit_gate,
        final_norm=final_norm,
        fashion_score=fashion_score,
        fashion_tip=fashion_tip,
        hard_veto_triggered=bool(fit_n < 0.30),
    )

    body_shape = body_vector.get("body_shape", "unknown") if body_vector else "unknown"
    improvements = build_improvement_suggestions(
        total=total,
        fit_n=fit_n,
        harmony_n=harmony_n,
        grooming_n=grooming_n,
        coherence_n=coherence_n,
        fashion_score=fashion_score,
        body_shape=body_shape,
        fit_tip=fit_tip,
        harmony_tip=harmony_tip,
        groom_tip=groom_tip,
        coherence_tip=coher_tip,
        fashion_tip=fashion_tip,
        num_fashion_matches=len(fashion_matches) if fashion_matches else 0,
    )

    return {
        "total": total,
        "fit_proportion": fit_score,
        "color_harmony": harmony_score,
        "color_season": season_score,
        "grooming": groom_score,
        "style_coherence": coher_score,
        "fashion_retrieval": fashion_score,
        "outfit_match_delta": outfit_delta_diag,
        "score_calculation": score_calculation,
        "why_this_score": why_this_score,
        "improvements": improvements,
        "body_shape": body_shape,
        "skin_season": skin_data.get("skin_season", "unknown") if skin_data else "unknown",
        "skin_hex": skin_data.get("hex", "#888888") if skin_data else "#888888",
        "breakdown": {
            "fit": fit_tip,
            "harmony": harmony_tip,
            "season": season_tip,
            "grooming": groom_tip,
            "coherence": coher_tip,
            "fashion_retrieval": fashion_tip,
            "match_delta": outfit_delta_tip,
        },
    }
