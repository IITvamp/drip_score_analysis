"""
mediapipe_compat.py
-------------------
Compatibility helpers for MediaPipe "solutions" API.
"""

from __future__ import annotations

import mediapipe as mp


def get_mediapipe_solutions():
    """
    Return the legacy `mp.solutions` namespace if available.
    Raise a clear runtime error otherwise.
    """
    solutions = getattr(mp, "solutions", None)
    if solutions is not None:
        return solutions

    raise RuntimeError(
        "This environment has a MediaPipe build without `mp.solutions`.\n"
        "The current codebase depends on MediaPipe Solutions APIs "
        "(Pose, FaceMesh, SelfieSegmentation).\n"
        "Use Python 3.10-3.12 and reinstall deps, for example:\n"
        "  pyenv local 3.11.11\n"
        "  python -m venv venv && source venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
    )
