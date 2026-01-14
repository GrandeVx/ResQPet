"""
Stray Index Calculator - Fusion Module
Combines all classifier outputs into a single Stray Index score
"""

import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from app.config import FUSION_WEIGHTS, STRAY_THRESHOLDS, MODELS
from app.models.backbone import DogPoseBackbone
from app.models.collar import CollarDetector
from app.models.skin import SkinClassifier
from app.models.pose import StrayPoseClassifier
from app.models.breed import BreedClassifier


class StrayStatus(Enum):
    """Classification status based on Stray Index"""
    OWNED = "owned"
    POSSIBLY_LOST = "possibly_lost"
    LIKELY_STRAY = "likely_stray"


@dataclass
class Detection:
    """Detection result for a single dog"""
    bbox: List[float]
    confidence: float
    stray_index: float
    status: StrayStatus
    components: Dict[str, float]
    breed_info: Optional[Dict] = None
    keypoints: Optional[np.ndarray] = None


class StrayIndexCalculator:
    """
    Main fusion module that combines all classifier outputs.

    Pipeline:
    1. Backbone detects dogs and extracts keypoints
    2. Each classifier produces a probability
    3. Weighted fusion produces Stray Index

    Stray Index Interpretation:
    - < 0.3: Likely owned (padronale)
    - 0.3 - 0.7: Possibly lost (possibile smarrito)
    - > 0.7: Likely stray (probabile randagio)
    """

    def __init__(self,
                 backbone_path: Optional[str] = None,
                 collar_path: Optional[str] = None,
                 skin_path: Optional[str] = None,
                 pose_path: Optional[str] = None,
                 breed_path: Optional[str] = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize all models.

        Args:
            *_path: Paths to model weights (None = use defaults or heuristics)
            weights: Custom fusion weights
        """
        # Load models
        self.backbone = DogPoseBackbone(
            model_path=backbone_path or str(MODELS.get('backbone', ''))
        )

        self.collar_detector = CollarDetector(
            model_path=collar_path or str(MODELS.get('collar', ''))
        )

        self.skin_classifier = SkinClassifier(
            model_path=skin_path or str(MODELS.get('skin', ''))
        )

        self.pose_classifier = StrayPoseClassifier(
            model_path=pose_path or str(MODELS.get('pose', ''))
        )

        self.breed_classifier = BreedClassifier(
            model_path=breed_path or str(MODELS.get('breed', ''))
        )

        # Fusion weights
        self.weights = weights or FUSION_WEIGHTS.copy()

        # Ensure weights sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

        print(f"[StrayIndexCalculator] Initialized with weights: {self.weights}")

    def compute_stray_index(self, frame: np.ndarray) -> List[Detection]:
        """
        Compute Stray Index for all dogs in frame.

        Args:
            frame: BGR image

        Returns:
            List of Detection objects for each dog found
        """
        # Step 1: Detect dogs and extract keypoints
        detections = self.backbone.detect(frame)

        if not detections:
            return []

        results = []

        for det in detections:
            roi = det['roi']
            keypoints = det.get('keypoints')
            bbox = det['bbox']
            det_confidence = det['confidence']

            # Step 2: Run each classifier
            # Collar detection
            p_no_collar = self.collar_detector.predict(roi)

            # Skin disease detection
            p_disease = self.skin_classifier.predict(roi)

            # Pose classification (if keypoints available)
            if keypoints is not None:
                normalized_kpts = self.backbone.normalize_keypoints(keypoints, bbox)
                p_stray_pose = self.pose_classifier.predict(normalized_kpts)
            else:
                p_stray_pose = 0.5  # Uncertain

            # Breed classification
            p_stray_breed = self.breed_classifier.get_stray_probability(roi)
            breed_info = self.breed_classifier.predict_with_details(roi)

            # Step 3: Weighted fusion
            components = {
                'collar': p_no_collar,
                'skin': p_disease,
                'pose': p_stray_pose,
                'breed': p_stray_breed
            }

            stray_index = self._fuse(components)

            # Step 4: Classify
            status = self._classify(stray_index)

            results.append(Detection(
                bbox=bbox,
                confidence=det_confidence,
                stray_index=stray_index,
                status=status,
                components=components,
                breed_info=breed_info,
                keypoints=keypoints
            ))

        return results

    def _fuse(self, components: Dict[str, float]) -> float:
        """
        Weighted fusion of component probabilities.

        Args:
            components: Dictionary of component probabilities

        Returns:
            Stray Index in [0, 1]
        """
        stray_index = 0.0

        for component, probability in components.items():
            weight = self.weights.get(component, 0.0)
            stray_index += probability * weight

        return np.clip(stray_index, 0.0, 1.0)

    def _classify(self, stray_index: float) -> StrayStatus:
        """
        Classify based on Stray Index thresholds.

        Args:
            stray_index: Computed index

        Returns:
            StrayStatus enum value
        """
        if stray_index < STRAY_THRESHOLDS['owned']:
            return StrayStatus.OWNED
        elif stray_index < STRAY_THRESHOLDS['lost']:
            return StrayStatus.POSSIBLY_LOST
        else:
            return StrayStatus.LIKELY_STRAY

    def analyze_image(self, image_path: str) -> Dict:
        """
        Analyze a single image file.

        Args:
            image_path: Path to image file

        Returns:
            Analysis results dictionary
        """
        frame = cv2.imread(image_path)
        if frame is None:
            return {'error': 'Failed to load image', 'detections': []}

        detections = self.compute_stray_index(frame)

        return {
            'image_path': image_path,
            'num_detections': len(detections),
            'detections': [self._detection_to_dict(d) for d in detections]
        }

    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single frame (for video processing).

        Args:
            frame: BGR image

        Returns:
            Analysis results dictionary
        """
        detections = self.compute_stray_index(frame)

        return {
            'num_detections': len(detections),
            'detections': [self._detection_to_dict(d) for d in detections]
        }

    def _detection_to_dict(self, detection: Detection) -> Dict:
        """Convert Detection to JSON-serializable dict"""
        return {
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'stray_index': round(detection.stray_index, 3),
            'status': detection.status.value,
            'status_color': self._get_status_color(detection.status),
            'components': {k: round(v, 3) for k, v in detection.components.items()},
            'breed': detection.breed_info.get('breed') if detection.breed_info else None,
            'breed_confidence': detection.breed_info.get('confidence') if detection.breed_info else None
        }

    def _get_status_color(self, status: StrayStatus) -> str:
        """Get color for status visualization"""
        colors = {
            StrayStatus.OWNED: '#22c55e',  # Green
            StrayStatus.POSSIBLY_LOST: '#eab308',  # Yellow
            StrayStatus.LIKELY_STRAY: '#ef4444'  # Red
        }
        return colors.get(status, '#6b7280')

    def draw_results(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """
        Draw detection results on frame.

        Args:
            frame: Original frame
            detections: List of detections

        Returns:
            Annotated frame
        """
        output = frame.copy()

        for det in detections:
            bbox = det.bbox
            stray_index = det.stray_index
            status = det.status

            # Color based on status
            color_map = {
                StrayStatus.OWNED: (0, 255, 0),  # Green
                StrayStatus.POSSIBLY_LOST: (0, 255, 255),  # Yellow
                StrayStatus.LIKELY_STRAY: (0, 0, 255)  # Red
            }
            color = color_map.get(status, (128, 128, 128))

            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"SI: {stray_index:.2f} ({status.value})"
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)

            # Draw label text
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw component breakdown
            y_offset = y2 + 15
            for comp_name, comp_value in det.components.items():
                comp_label = f"{comp_name}: {comp_value:.2f}"
                cv2.putText(output, comp_label, (x1, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
                y_offset += 12

        return output


# Convenience function for quick analysis
def analyze_image(image_path: str) -> Dict:
    """Quick image analysis with default settings"""
    calculator = StrayIndexCalculator()
    return calculator.analyze_image(image_path)
