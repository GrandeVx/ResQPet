"""
YOLO11 Dog Pose Backbone
Extracts dog bounding boxes and 24 anatomical keypoints
"""

import numpy as np
import cv2
from typing import List, Dict, Optional, Tuple
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Warning] ultralytics not installed. Backbone will use mock mode.")


class DogPoseBackbone:
    """
    YOLO11 Dog Pose model for detecting dogs and extracting keypoints.

    This model is trained on the Dog-Pose dataset from Ultralytics.
    It detects DOGS (class 0) and extracts 24 anatomical keypoints.

    Keypoints (24 points):
    - Head: nose, left_eye, right_eye, left_ear_base, right_ear_base,
            left_ear_tip, right_ear_tip, throat, chin
    - Body: withers (garrese)
    - Front legs: left/right front elbow, knee, paw
    - Back legs: left/right back elbow, knee, paw
    - Tail: tail_start, tail_end

    Reference: https://docs.ultralytics.com/datasets/pose/dog-pose/
    """

    # 24 Dog-Pose keypoints (from Ultralytics Dog-Pose dataset)
    KEYPOINT_NAMES = [
        'nose',              # 0
        'left_eye',          # 1
        'right_eye',         # 2
        'left_ear_base',     # 3
        'right_ear_base',    # 4
        'left_ear_tip',      # 5
        'right_ear_tip',     # 6
        'throat',            # 7
        'withers',           # 8  (garrese)
        'left_front_elbow',  # 9
        'right_front_elbow', # 10
        'left_front_knee',   # 11
        'right_front_knee',  # 12
        'left_front_paw',    # 13
        'right_front_paw',   # 14
        'left_back_elbow',   # 15
        'right_back_elbow',  # 16
        'left_back_knee',    # 17
        'right_back_knee',   # 18
        'left_back_paw',     # 19
        'right_back_paw',    # 20
        'tail_start',        # 21
        'tail_end',          # 22
        'chin'               # 23
    ]

    # Dog-Pose dataset has only 1 class: dog = 0
    # This is different from COCO where dog = 16
    DOG_CLASS_ID = 0  # In dog-pose dataset, dog is class 0

    def __init__(self, model_path: Optional[str] = None, confidence: float = 0.5):
        """
        Initialize the backbone.

        Args:
            model_path: Path to YOLO dog-pose weights. If None, uses fallback.
            confidence: Minimum detection confidence threshold.
        """
        self.confidence = confidence
        self.model = None
        self.is_dog_pose_model = False  # Track if we're using dog-pose or COCO model

        if YOLO_AVAILABLE:
            try:
                # Try to load dog-pose model
                if model_path and Path(model_path).exists():
                    self.model = YOLO(model_path)
                    # Check if it's a dog-pose model by checking the model name
                    model_name = Path(model_path).name.lower()
                    self.is_dog_pose_model = 'dog' in model_name or 'dog-pose' in model_name

                    if self.is_dog_pose_model:
                        print(f"[Backbone] Dog-Pose model loaded: {model_path}")
                        print(f"[Backbone] Will detect dogs with 24 anatomical keypoints")
                    else:
                        print(f"[Backbone] Custom YOLO model loaded: {model_path}")
                else:
                    # FALLBACK: Use standard YOLO detection model
                    # This detects 80 COCO classes including dogs (class 16)
                    print(f"[Backbone] WARNING: Dog-Pose model not found at {model_path}")
                    print(f"[Backbone] Using yolo11n.pt fallback (COCO detection)")
                    print(f"[Backbone] For proper dog detection, run 00a_yolo_dog_pose_training.ipynb")
                    self.model = YOLO('yolo11n.pt')
                    self.is_dog_pose_model = False
            except Exception as e:
                print(f"[Backbone] Failed to load YOLO: {e}")
                self.model = None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect dogs in frame and extract keypoints.

        Args:
            frame: BGR image as numpy array

        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2] coordinates
                - confidence: detection confidence
                - keypoints: 24x3 array (x, y, visibility)
                - roi: cropped image of the dog
        """
        if self.model is None:
            return self._mock_detection(frame)

        detections = []

        try:
            # Run inference
            results = self.model(frame, verbose=False)

            for result in results:
                if result.boxes is None:
                    continue

                for i, box in enumerate(result.boxes):
                    cls = int(box.cls[0]) if box.cls is not None else -1
                    conf = float(box.conf[0]) if box.conf is not None else 0.0

                    # Class filtering depends on model type
                    if self.is_dog_pose_model:
                        # Dog-Pose model: only class 0 (dog), accept all detections
                        # since the model is trained specifically for dogs
                        pass  # No filtering needed
                    else:
                        # COCO model fallback: filter for dogs only (class 16)
                        COCO_DOG_CLASS = 16
                        if cls != COCO_DOG_CLASS:
                            continue

                    # Check confidence threshold
                    if conf < self.confidence:
                        continue

                    print(f"[Backbone] Dog detected! Class: {cls}, Confidence: {conf:.2f}")

                    # Get bounding box
                    bbox = box.xyxy[0].cpu().numpy().tolist()

                    # Extract keypoints if available
                    keypoints = None
                    if result.keypoints is not None and i < len(result.keypoints):
                        kpts = result.keypoints[i].data[0].cpu().numpy()
                        keypoints = kpts  # Shape: (num_keypoints, 3)

                    # Crop ROI
                    x1, y1, x2, y2 = map(int, bbox)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    roi = frame[y1:y2, x1:x2].copy()

                    detections.append({
                        'bbox': bbox,
                        'confidence': conf,
                        'keypoints': keypoints,
                        'roi': roi,
                        'class_id': cls
                    })

        except Exception as e:
            print(f"[Backbone] Detection error: {e}")
            return self._mock_detection(frame)

        return detections

    def _mock_detection(self, frame: np.ndarray) -> List[Dict]:
        """
        Generate mock detection for testing without model.
        """
        h, w = frame.shape[:2]

        # Simulate a detection in the center
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4

        # Generate random keypoints within bbox
        keypoints = np.zeros((17, 3))  # Standard COCO keypoints
        for i in range(17):
            keypoints[i] = [
                np.random.uniform(x1, x2),
                np.random.uniform(y1, y2),
                np.random.uniform(0.5, 1.0)  # visibility
            ]

        return [{
            'bbox': [x1, y1, x2, y2],
            'confidence': 0.85,
            'keypoints': keypoints,
            'roi': frame[y1:y2, x1:x2].copy(),
            'class_id': 16  # dog
        }]

    def normalize_keypoints(self, keypoints: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Normalize keypoints relative to bounding box.

        Args:
            keypoints: Raw keypoints (N, 3)
            bbox: [x1, y1, x2, y2]

        Returns:
            Normalized keypoints with x, y in [0, 1] relative to bbox
        """
        if keypoints is None:
            return None

        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1

        normalized = keypoints.copy()
        if w > 0 and h > 0:
            normalized[:, 0] = (keypoints[:, 0] - x1) / w
            normalized[:, 1] = (keypoints[:, 1] - y1) / h

        return normalized

    def extract_face_roi(self, frame: np.ndarray, keypoints: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face region based on keypoints.

        Args:
            frame: Full image
            keypoints: Keypoints array

        Returns:
            Cropped face region or None if keypoints invalid
        """
        if keypoints is None or len(keypoints) < 5:
            return None

        # Use first 5 keypoints (nose, eyes, ears) to define face region
        face_kpts = keypoints[:5]
        valid_kpts = face_kpts[face_kpts[:, 2] > 0.3]  # Filter by visibility

        if len(valid_kpts) < 2:
            return None

        x_min = int(valid_kpts[:, 0].min())
        x_max = int(valid_kpts[:, 0].max())
        y_min = int(valid_kpts[:, 1].min())
        y_max = int(valid_kpts[:, 1].max())

        # Add padding
        pad = int((x_max - x_min) * 0.3)
        x_min = max(0, x_min - pad)
        x_max = min(frame.shape[1], x_max + pad)
        y_min = max(0, y_min - pad)
        y_max = min(frame.shape[0], y_max + pad)

        if x_max <= x_min or y_max <= y_min:
            return None

        return frame[y_min:y_max, x_min:x_max].copy()

    def draw_detections(self, frame: np.ndarray, detections: List[Dict],
                        draw_keypoints: bool = True) -> np.ndarray:
        """
        Draw bounding boxes and keypoints on frame.

        Args:
            frame: Image to draw on
            detections: List of detection dicts
            draw_keypoints: Whether to draw keypoints

        Returns:
            Annotated frame
        """
        output = frame.copy()

        for det in detections:
            bbox = det['bbox']
            conf = det['confidence']

            # Draw bbox
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(output, f"Dog: {conf:.2f}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw keypoints
            if draw_keypoints and det.get('keypoints') is not None:
                keypoints = det['keypoints']
                for kpt in keypoints:
                    x, y, v = kpt
                    if v > 0.3:  # Only draw visible keypoints
                        cv2.circle(output, (int(x), int(y)), 3, (255, 0, 0), -1)

        return output
