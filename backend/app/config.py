"""
Configuration settings for ResQPet
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
DATA_DIR = PROJECT_ROOT / 'data'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'

# Model paths
# NOTE: backbone uses yolo11n-dog-pose-v2.pt trained on Dog-Pose dataset (150 epochs)
# Metriche: mAP@50=98.7%, Precision=96.9%, Recall=97.7%
# This model detects DOGS (not humans) and extracts 24 anatomical keypoints
MODELS = {
    'backbone': WEIGHTS_DIR / 'yolo11n-dog-pose-v2.pt',  # Dog detection with 24 keypoints (v2)
    'collar': WEIGHTS_DIR / 'collar_detector.pt',
    'skin': WEIGHTS_DIR / 'skin_classifier.pt',
    'pose': WEIGHTS_DIR / 'stray_pose_classifier.pt',
    'breed': WEIGHTS_DIR / 'breed_classifier.pt',
}

# Breed priors lookup table
BREED_PRIORS_PATH = DATA_DIR / 'breed_priors.json'

# Fusion weights
FUSION_WEIGHTS = {
    'collar': 0.35,
    'skin': 0.20,
    'pose': 0.25,
    'breed': 0.20
}

# Stray Index thresholds
STRAY_THRESHOLDS = {
    'owned': 0.3,      # < 0.3 = Padronale
    'lost': 0.7,       # 0.3 - 0.7 = Possibile Smarrito
    # > 0.7 = Probabile Randagio
}

# Detection settings
DETECTION_CONFIDENCE = 0.5
KEYPOINT_CONFIDENCE = 0.7

# Video processing
MAX_FPS = 15
FRAME_BUFFER_SIZE = 30

# Alert settings
ALERT_COOLDOWN_SECONDS = 30  # Tempo minimo tra alert per stesso cane
