"""
color_theory.py
---------------
Seasonal color analysis and outfit color harmony scoring.
All color analysis is done in CIE LAB colorspace for perceptual uniformity.
"""

import math
import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


# ------------------------------------------------------------------
# Seasonal palette definitions in LAB space
# ------------------------------------------------------------------

SEASONAL_PALETTES = {
    "spring": {
        "recommended": [
            {"L": (60, 95), "a": (5, 28), "b": (8, 42)},    # peach / coral
            {"L": (45, 72), "a": (-18, 4), "b": (14, 40)},   # warm olive / camel
            {"L": (70, 95), "a": (0, 15), "b": (15, 38)},    # warm ivory / cream
        ],
        "avoid": [
            {"L": (0, 38), "a": (-4, 4), "b": (-4, 4)},      # black / charcoal
            {"L": (50, 75), "a": (-20, -8), "b": (-30, -10)}, # cool grey-blue
        ],
        "description": "Warm peachy neutrals, coral, ivory, warm greens, camel",
        "best_colors": ["coral", "peach", "warm white", "camel", "light warm green"],
        "avoid_colors": ["black", "cool grey", "icy blue"],
    },
    "summer": {
        "recommended": [
            {"L": (48, 88), "a": (-12, 10), "b": (-34, -2)},  # lavender / soft blue
            {"L": (60, 88), "a": (5, 22), "b": (-8, 12)},     # rose / soft pink
            {"L": (70, 92), "a": (-8, 6), "b": (-12, 8)},     # powder / soft neutral
        ],
        "avoid": [
            {"L": (38, 68), "a": (10, 32), "b": (20, 52)},    # orange / rust
            {"L": (30, 55), "a": (8, 25), "b": (18, 42)},     # brown / warm tan
        ],
        "description": "Soft blues, lavender, rose, powder pink, slate grey",
        "best_colors": ["lavender", "soft blue", "rose", "mauve", "powder pink"],
        "avoid_colors": ["orange", "rust", "warm brown", "mustard"],
    },
    "autumn": {
        "recommended": [
            {"L": (22, 55), "a": (10, 36), "b": (18, 52)},    # rust / terracotta
            {"L": (28, 58), "a": (-12, 10), "b": (18, 46)},   # olive / khaki
            {"L": (35, 65), "a": (5, 22), "b": (22, 48)},     # mustard / warm tan
        ],
        "avoid": [
            {"L": (68, 100), "a": (-5, 5), "b": (-22, 0)},    # icy pastels
            {"L": (55, 82), "a": (-4, 8), "b": (-28, -10)},   # cool blue-grey
        ],
        "description": "Rust, terracotta, olive, mustard, warm brown, forest green",
        "best_colors": ["rust", "terracotta", "olive", "mustard", "burnt orange"],
        "avoid_colors": ["icy pastels", "cool blue", "fuchsia"],
    },
    "winter": {
        "recommended": [
            {"L": (0, 28), "a": (-5, 5), "b": (-6, 6)},       # true black
            {"L": (85, 100), "a": (-5, 5), "b": (-6, 6)},     # pure white
            {"L": (28, 58), "a": (-12, 4), "b": (-42, -12)},  # royal blue / navy
            {"L": (30, 55), "a": (15, 40), "b": (-18, 8)},    # jewel tones / fuchsia
        ],
        "avoid": [
            {"L": (40, 68), "a": (8, 28), "b": (18, 46)},     # earth tones
            {"L": (60, 82), "a": (2, 18), "b": (12, 36)},     # warm beige / tan
        ],
        "description": "True black, white, royal blue, jewel tones, icy clear colors",
        "best_colors": ["black", "white", "royal blue", "fuchsia", "icy pink"],
        "avoid_colors": ["earth tones", "warm beige", "orange-red", "mustard"],
    },
}


# ------------------------------------------------------------------
# RGB ↔ LAB conversions
# ------------------------------------------------------------------

def rgb_to_lab(r: int, g: int, b: int) -> tuple[float, float, float]:
    """Convert 0-255 RGB to CIE LAB."""
    rgb_color = sRGBColor(r / 255.0, g / 255.0, b / 255.0)
    lab_color = convert_color(rgb_color, LabColor)
    return (lab_color.lab_l, lab_color.lab_a, lab_color.lab_b)


def lab_to_hue_degrees(L: float, a: float, b: float) -> float:
    """Convert LAB to hue angle 0-360 using atan2(b, a)."""
    hue = math.degrees(math.atan2(b, a))
    return hue % 360


def color_chroma(a: float, b: float) -> float:
    """Chroma (saturation) in LAB."""
    return math.sqrt(a * a + b * b)


# ------------------------------------------------------------------
# Skin season classifier
# ------------------------------------------------------------------

def classify_skin_season(L: float, a: float, b: float) -> str:
    """
    Classify skin into one of four color seasons using LAB values.
    L=lightness (0-100), a=green-red axis, b=blue-yellow axis.
    """
    warm = a > 7.0          # positive a = warm/reddish undertone
    light = L > 58.0        # higher L = lighter skin

    if light and warm:
        return "spring"
    elif light and not warm:
        return "summer"
    elif not light and warm:
        return "autumn"
    else:
        return "winter"


def _color_in_range(L: float, a: float, b: float, lab_range: dict) -> bool:
    lL, lA, lB = lab_range["L"], lab_range["a"], lab_range["b"]
    return (lL[0] <= L <= lL[1]) and (lA[0] <= a <= lA[1]) and (lB[0] <= b <= lB[1])


# ------------------------------------------------------------------
# Color season match scoring (out of 20 pts)
# ------------------------------------------------------------------

def score_color_season_match(clothing_lab_colors: list[tuple], skin_season: str) -> tuple[int, str]:
    """
    Score how well clothing colors match the seasonal palette.
    Returns (score 0-20, explanation string).
    """
    if skin_season not in SEASONAL_PALETTES:
        return 10, "Unknown season — neutral score"

    palette = SEASONAL_PALETTES[skin_season]
    rec_ranges = palette["recommended"]
    avoid_ranges = palette["avoid"]

    bonus = 0
    all_good = True
    matched_names = []
    penalized_names = []

    for lab in clothing_lab_colors:
        L, a, b = lab
        in_rec = any(_color_in_range(L, a, b, r) for r in rec_ranges)
        in_avoid = any(_color_in_range(L, a, b, r) for r in avoid_ranges)

        if in_rec:
            bonus += 7
            matched_names.append(f"L={L:.0f}")
        elif in_avoid:
            bonus -= 6
            all_good = False
            penalized_names.append(f"L={L:.0f}")
        else:
            all_good = False

    if all_good and len(clothing_lab_colors) >= 2:
        bonus += 6  # all colors match

    raw = max(0, min(20, bonus + 7))  # base of 7

    tips = palette["best_colors"][:3]
    tip_str = ", ".join(tips)

    if matched_names:
        explanation = f"Colors suit your {skin_season} palette. Try: {tip_str}"
    elif penalized_names:
        explanation = f"Some colors clash with your {skin_season} tone. Better options: {tip_str}"
    else:
        explanation = f"Neutral palette match. Optimal choices for {skin_season}: {tip_str}"

    return int(raw), explanation


# ------------------------------------------------------------------
# Color harmony scoring (out of 25 pts)
# ------------------------------------------------------------------

def score_color_harmony(
    torso_colors_lab: list[tuple],
    lower_colors_lab: list[tuple],
    num_distinct: int,
) -> tuple[int, str]:
    """
    Score outfit color harmony between top and bottom garments.
    Returns (score 0-25, explanation).
    """
    if not torso_colors_lab or not lower_colors_lab:
        return 12, "Could not detect outfit colors clearly"

    # Use dominant color from each region
    t_L, t_a, t_b = torso_colors_lab[0]
    l_L, l_a, l_b = lower_colors_lab[0]

    t_hue = lab_to_hue_degrees(t_L, t_a, t_b)
    l_hue = lab_to_hue_degrees(l_L, l_a, l_b)
    t_chroma = color_chroma(t_a, t_b)
    l_chroma = color_chroma(l_a, l_b)

    hue_diff = abs(t_hue - l_hue)
    if hue_diff > 180:
        hue_diff = 360 - hue_diff

    # Lightness contrast
    lightness_diff = abs(t_L - l_L)

    # Penalty for too many colors
    color_penalty = max(0, (num_distinct - 3) * 4)

    # Classify relationship
    if t_chroma < 12 or l_chroma < 12:
        # One or both items are neutral (grey/white/black/beige)
        if lightness_diff > 25:
            score = 23
            explanation = "Neutral + contrast — clean and versatile"
        else:
            score = 18
            explanation = "All-neutral — safe, add one accent piece"
    elif hue_diff <= 20:
        # Monochromatic
        if lightness_diff > 20:
            score = 25
            explanation = "Monochromatic with tonal contrast — sophisticated"
        else:
            score = 20
            explanation = "Monochromatic — try varying lightness for depth"
    elif 150 <= hue_diff <= 210:
        # Complementary
        score = 22
        explanation = "Complementary colors — bold and intentional"
    elif hue_diff <= 50:
        # Analogous
        score = 19
        explanation = "Analogous palette — cohesive and easy on the eye"
    elif 110 <= hue_diff <= 150:
        # Split complementary / triadic
        score = 15
        explanation = "Triadic pairing — works if executed deliberately"
    else:
        # Random
        score = 8
        explanation = "Colors don't relate — consider a neutral base"

    final = max(0, min(25, score - color_penalty))
    if color_penalty > 0:
        explanation += f" (−{color_penalty} pts: {num_distinct} colors is too busy)"

    return int(final), explanation
