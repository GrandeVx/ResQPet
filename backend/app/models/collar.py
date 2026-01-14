"""
Collar Detector
Detects presence/absence of collar, harness, or leash on dogs
"""

import numpy as np
import cv2
from typing import Dict, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class CollarDetector:
    """
    YOLOv8n-based detector for collar/harness/leash detection.

    Classes:
        - collar: Visible collar
        - harness: Harness/pettorina
        - leash: Leash attached

    Output: P(no_collar) - probability that dog has NO collar
    """

    CLASSES = ['Dog-with-Leash', 'Dog-without-Leash']

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.5):
        """
        Initialize collar detector.

        Args:
            model_path: Path to trained YOLOv8 weights
            confidence: Detection confidence threshold
        """
        self.confidence = confidence
        self.model = None

        if YOLO_AVAILABLE and model_path and Path(model_path).exists():
            try:
                self.model = YOLO(model_path)
                print(f"[CollarDetector] Model loaded from {model_path}")
            except Exception as e:
                print(f"[CollarDetector] Failed to load model: {e}")

    def predict(self, roi: np.ndarray) -> float:
        """
        Predict probability of NO collar on the dog.

        Args:
            roi: Cropped image of the dog (BGR)

        Returns:
            P(no_collar) in [0, 1]
            Higher value = more likely NO collar present
        """
        if roi is None or roi.size == 0:
            return 0.8  # Default: assume no collar if invalid input

        if self.model is None:
            return self._heuristic_detection(roi)

        try:
            # Run YOLO detection
            results = self.model(roi, verbose=False)

            max_collar_conf = 0.0
            has_leash = False

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    # Check class name
                    class_name = self.CLASSES[cls] if cls < len(self.CLASSES) else ''

                    if 'with-Leash' in class_name or 'collar' in class_name.lower():
                        max_collar_conf = max(max_collar_conf, conf)
                        if conf > self.confidence:
                            has_leash = True

            # Calculate P(no_collar)
            if has_leash:
                return 1.0 - max_collar_conf  # Low probability of no collar
            else:
                return 0.8  # High probability of no collar

        except Exception as e:
            print(f"[CollarDetector] Prediction error: {e}")
            return self._heuristic_detection(roi)

    def _heuristic_detection(self, roi: np.ndarray) -> float:
        """
        Heuristic collar detection using color analysis.
        Looks for collar-like patterns in the neck region.

        Args:
            roi: Dog image

        Returns:
            P(no_collar) estimate
        """
        if roi is None or roi.size == 0:
            return 0.8

        try:
            h, w = roi.shape[:2]

            # Focus on neck region (upper-middle part of dog)
            neck_y_start = int(h * 0.1)
            neck_y_end = int(h * 0.35)
            neck_x_start = int(w * 0.25)
            neck_x_end = int(w * 0.75)

            neck_region = roi[neck_y_start:neck_y_end, neck_x_start:neck_x_end]

            if neck_region.size == 0:
                return 0.7

            # Convert to HSV
            hsv = cv2.cvtColor(neck_region, cv2.COLOR_BGR2HSV)

            # Collars are often bright/saturated colors
            # Look for high saturation regions (artificial colors)
            saturation = hsv[:, :, 1]
            high_sat_pixels = np.sum(saturation > 100)
            total_pixels = saturation.size

            sat_ratio = high_sat_pixels / total_pixels if total_pixels > 0 else 0

            # If high saturation detected, likely collar present
            if sat_ratio > 0.15:
                return 0.3  # Likely has collar
            elif sat_ratio > 0.05:
                return 0.5  # Uncertain
            else:
                return 0.75  # Probably no collar

        except Exception as e:
            print(f"[CollarDetector] Heuristic error: {e}")
            return 0.7

    def predict_with_details(self, roi: np.ndarray) -> Dict:
        """
        Predict with detailed breakdown.

        Args:
            roi: Dog image

        Returns:
            Dictionary with detailed predictions
        """
        p_no_collar = self.predict(roi)

        return {
            'p_no_collar': p_no_collar,
            'p_has_collar': 1.0 - p_no_collar,
            'collar_detected': p_no_collar < 0.5,
            'confidence': abs(p_no_collar - 0.5) * 2  # Confidence in prediction
        }
