"""
Configuration for ResQPet Labeling Tool
"""

import os
from pathlib import Path

# Base paths
LABELING_TOOL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LABELING_TOOL_DIR.parent
LABELING_DATA_DIR = PROJECT_ROOT / 'labeling_data'

# Database
DATABASE_PATH = LABELING_DATA_DIR / 'labeling.db'
SQLALCHEMY_DATABASE_URI = f'sqlite:///{DATABASE_PATH}'

# Dataset paths
DATASETS = {
    'dog-pose': PROJECT_ROOT / 'datasets' / 'dog-pose',
    'dog-with-leash': PROJECT_ROOT / 'datasets' / 'dog-with-leash',
    'dog-skin-diseases': PROJECT_ROOT / 'datasets' / 'dog-skin-diseases',
    'stanford-dogs': PROJECT_ROOT / 'datasets' / 'stanford-dogs',
    'stray-dogs-fyp': PROJECT_ROOT / 'datasets' / 'stray-dogs-fyp',
}

# Model paths
WEIGHTS_DIR = PROJECT_ROOT / 'weights'
BACKBONE_MODEL = WEIGHTS_DIR / 'yolo11n-dog-pose-v2.pt'  # 150 epochs, mAP@50=98.7%
COLLAR_MODEL = WEIGHTS_DIR / 'collar_detector.pt'

# Export settings
EXPORTS_DIR = LABELING_DATA_DIR / 'exports'

# Image settings
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
MAX_IMAGE_SIZE = 1920  # Max dimension for display

# Flask settings
SECRET_KEY = os.environ.get('SECRET_KEY', 'resqpet-labeling-dev-key')
DEBUG = os.environ.get('FLASK_DEBUG', 'true').lower() == 'true'

# Labeling settings
DEFAULT_BATCH_SIZE = 100
PRELABEL_CONFIDENCE_THRESHOLD = 0.5

# Multi-user configuration
USERS = {
    1: "Utente 1",
    2: "Utente 2",
    3: "Utente 3",
    4: "Utente 4",
    5: "Utente 5",
}
NUM_USERS = 5
