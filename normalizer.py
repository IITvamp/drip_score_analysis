"""
normalizer.py
-------------
Perspective-invariant measurement system.
All pixel distances are divided by inter-ocular distance (face unit) so that
the same person at 1 m and at 3 m yields identical body ratios.

Standalone test:
    python normalizer.py
Opens webcam and prints the face-unit value in the corner.
Confirm it stays within ±5 px as you step closer / farther.
"""

import math
import cv2
import numpy as np


class PerspectiveNormalizer:
    # MediaPipe Face Mesh outer eye-corner indices
    LEFT_EYE_OUTER = 33
    RIGHT_EYE_OUTER = 263

    # Typical adult face-unit reference ranges (in face units)
    TYPICAL_SHOULDER_FU = (3.2, 4.8)
    TYPICAL_HIP_FU = (2.8, 4.2)
    TYPICAL_TORSO_FU = (4.0, 6.0)

    def __init__(self):
        self.face_unit_px: float | None = None
        self.frame_w: int | None = None
        self.frame_h: int | None = None

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------

    def calibrate(self, face_landmarks, frame_w: int, frame_h: int) -> float | None:
        """
        Compute face unit from detected Face Mesh landmarks.
        Returns inter-ocular distance in px, or None if unreliable.
        """
        self.frame_w = frame_w
        self.frame_h = frame_h

        if face_landmarks is None:
            self.face_unit_px = None
            return None

        lm = face_landmarks.landmark
        left = lm[self.LEFT_EYE_OUTER]
        right = lm[self.RIGHT_EYE_OUTER]

        dx = (right.x - left.x) * frame_w
        dy = (right.y - left.y) * frame_h
        iod = math.sqrt(dx * dx + dy * dy)

        # Sanity: ignore implausible detections
        if iod < 15 or iod > 320:
            self.face_unit_px = None
            return None

        self.face_unit_px = iod
        return iod

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_face_units(self, pixel_distance: float) -> float:
        if not self.face_unit_px:
            raise RuntimeError("Normalizer not calibrated. Call calibrate() first.")
        return pixel_distance / self.face_unit_px

    def landmark_to_px(self, landmark, frame_w: int = None, frame_h: int = None) -> tuple[float, float]:
        w = frame_w or self.frame_w
        h = frame_h or self.frame_h
        return (landmark.x * w, landmark.y * h)

    def distance_px(self, lm1, lm2) -> float:
        p1 = self.landmark_to_px(lm1)
        p2 = self.landmark_to_px(lm2)
        return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)

    def distance_face_units(self, lm1, lm2) -> float:
        return self.to_face_units(self.distance_px(lm1, lm2))

    def is_valid(self) -> bool:
        return self.face_unit_px is not None

    def debug_info(self) -> dict:
        if self.face_unit_px is None:
            dist_label = "no face"
        elif self.face_unit_px > 90:
            dist_label = "close"
        elif self.face_unit_px > 50:
            dist_label = "medium"
        else:
            dist_label = "far"
        return {"face_unit_px": self.face_unit_px, "estimated_distance": dist_label}


# ------------------------------------------------------------------
# Standalone webcam test
# ------------------------------------------------------------------

if __name__ == "__main__":
    from mediapipe_compat import get_mediapipe_solutions

    try:
        mp_solutions = get_mediapipe_solutions()
    except RuntimeError as e:
        print(e)
        raise SystemExit(1)

    mp_face_mesh = mp_solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    normalizer = PerspectiveNormalizer()
    history = []

    print("Webcam test — move closer and farther. Face unit should stay stable.")
    print("Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        fu = None
        if results.multi_face_landmarks:
            fu = normalizer.calibrate(results.multi_face_landmarks[0], w, h)

        # Draw eye landmarks
        if fu and results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            for idx in [PerspectiveNormalizer.LEFT_EYE_OUTER, PerspectiveNormalizer.RIGHT_EYE_OUTER]:
                px = int(lm[idx].x * w)
                py = int(lm[idx].y * h)
                cv2.circle(frame, (px, py), 4, (0, 255, 0), -1)

        # Stats overlay
        if fu:
            history.append(fu)
            if len(history) > 30:
                history.pop(0)
            variance = np.std(history) if len(history) > 5 else 0.0
            cv2.putText(frame, f"Face unit: {fu:.1f} px", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Std (last 30): {variance:.2f} px", (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 200), 2)
            cv2.putText(frame, f"Status: {normalizer.debug_info()['estimated_distance']}", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 0), 2)
        else:
            cv2.putText(frame, "No face detected — face the camera", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "Press Q to quit", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        cv2.imshow("Normalizer Test", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()
