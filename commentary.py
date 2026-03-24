"""
commentary.py
-------------
Gen-Z office commentary mapped to score bands, body shapes, and seasons.
"""

import random

_BAND_COMMENTS = {
    (0, 30): [
        "HR wants a word. Like, several words.",
        "The fit is... a choice. A very brave choice.",
        "Did you get dressed in the dark? Asking for a friend.",
        "This outfit said 'I tried' but the clothes said 'no you didn't'.",
        "Bestie this is not it.",
    ],
    (31, 45): [
        "Business casual. But which business though?",
        "Solid C+ fit. Professor would let you present last.",
        "The fit is giving 'meeting could've been an email'.",
        "Not bad. Not good. Comfortably mediocre.",
        "You got dressed. Points for participation.",
    ],
    (46, 60): [
        "Mid-senior energy. Your fit has a LinkedIn.",
        "The outfit is coherent. You understood the dress code.",
        "B- energy. You read the memo.",
        "The clothes are there. They're doing something.",
        "Respectful. The outfit has entered the building.",
    ],
    (61, 75): [
        "Okay we see you. The confidence is WARRANTED.",
        "The fit is doing the networking for you.",
        "B+ fit. Your clothes are pulling their weight.",
        "The outfit just walked in like it knows the quarterly targets.",
        "Sharp. You didn't come to play.",
    ],
    (76, 85): [
        "A- energy. The boardroom is not ready.",
        "The fit is giving 'I make the decisions around here'.",
        "Lowkey elevated. The tailor clapped.",
        "Your outfit has a 5-year plan.",
        "You look expensive. Not in a bad way.",
    ],
    (86, 100): [
        "Certified Drip Lord. The office is not ready.",
        "You are THAT person. Congratulations.",
        "The fit said 'I woke up like this' and we believe it.",
        "Board meeting? Runway? Same difference at this point.",
        "The CEO is taking notes on YOUR fit. Slay.",
    ],
}

_SHAPE_TIPS = {
    "inverted_triangle": "V-shape appreciators have logged on.",
    "hourglass": "Proportions? Immaculate.",
    "rectangle": "Clean lines, clean fit. The geometry is doing work.",
    "triangle": "Base secured. Now build the statement on top.",
    "oval": "Comfort is a vibe. Own it confidently.",
}

_SEASON_TIPS = {
    "spring": "Try coral, peach, or warm ivory tones — they'll light you up.",
    "summer": "Soft lavender, dusty rose, or slate blue are your power palette.",
    "autumn": "Rust, olive, and mustard are literally made for your skin tone.",
    "winter": "Go bold — true black, white, or jewel tones are your lane.",
}


def get_commentary(score: int, body_shape: str, skin_season: str) -> dict:
    """
    Returns:
        main_line   : main roast/praise
        shape_line  : body-shape specific note
        color_tip   : one color recommendation
    """
    main_line = "The fit exists. We acknowledge it."
    for (lo, hi), lines in _BAND_COMMENTS.items():
        if lo <= score <= hi:
            main_line = random.choice(lines)
            break

    shape_line = _SHAPE_TIPS.get(body_shape, "Body detected. Proceed.")
    color_tip = _SEASON_TIPS.get(skin_season, "Experiment with your palette.")

    return {
        "main_line": main_line,
        "shape_line": shape_line,
        "color_tip": color_tip,
    }


def score_label(score: int) -> str:
    if score >= 86:
        return "DRIP LORD"
    elif score >= 76:
        return "ELEVATED"
    elif score >= 61:
        return "SHARP"
    elif score >= 46:
        return "DECENT"
    elif score >= 31:
        return "MEDIOCRE"
    else:
        return "NEEDS WORK"


def score_color_bgr(score: int) -> tuple:
    """OpenCV BGR color for the score."""
    if score >= 86:
        return (0, 215, 0)      # bright green
    elif score >= 76:
        return (0, 200, 100)    # green-teal
    elif score >= 61:
        return (0, 180, 230)    # yellow-orange
    elif score >= 46:
        return (0, 140, 255)    # orange
    elif score >= 31:
        return (60, 100, 255)   # orange-red
    else:
        return (0, 0, 220)      # red
