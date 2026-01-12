"""
Pre-labeling Service
Uses trained models to automatically generate initial labels for images
"""

import cv2
import sys
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from labeling_tool.database import db
from labeling_tool.database.models import Image, CollarAnnotation, LabelStatus, CollarLabel
from labeling_tool import config

# Add backend to path for model imports
backend_path = config.PROJECT_ROOT / 'backend'
if str(backend_path) not in sys.path:
    sys.path.insert(0, str(backend_path))


class PreLabeler:
    """Batch pre-labeling using trained collar detector."""

    def __init__(self, backbone_path: str = None, collar_path: str = None):
        """
        Initialize pre-labeler with model paths.

        Args:
            backbone_path: Path to backbone model (dog detection + keypoints)
            collar_path: Path to collar detector model
        """
        self.backbone_path = backbone_path or str(config.BACKBONE_MODEL)
        self.collar_path = collar_path or str(config.COLLAR_MODEL)

        self.backbone = None
        self.collar = None

        self._load_models()

    def _load_models(self):
        """Load the models."""
        try:
            from app.models.backbone import DogPoseBackbone
            from app.models.collar import CollarDetector

            print("[PreLabeler] Loading backbone model...")
            self.backbone = DogPoseBackbone(
                model_path=self.backbone_path,
                confidence=0.5
            )

            print("[PreLabeler] Loading collar detector...")
            self.collar = CollarDetector(
                model_path=self.collar_path,
                confidence=0.5
            )

            print("[PreLabeler] Models loaded successfully")

        except Exception as e:
            print(f"[PreLabeler] Error loading models: {e}")
            print("[PreLabeler] Will use heuristic fallback")

    def prelabel_all(self, batch_size=100, skip_existing=True):
        """
        Pre-label all unlabeled images.

        Args:
            batch_size: Commit after this many images
            skip_existing: Skip images that already have annotations

        Returns:
            Dict with processing statistics
        """
        # Query unlabeled images
        query = Image.query.filter(Image.status == LabelStatus.UNLABELED)

        if skip_existing:
            query = query.filter(Image.collar_label == None)

        total = query.count()
        print(f"[PreLabeler] {total} images to process")

        if total == 0:
            return {'processed': 0, 'errors': 0, 'total': 0}

        processed = 0
        errors = 0
        no_dog_count = 0

        for image in tqdm(query.yield_per(batch_size), total=total, desc="Pre-labeling"):
            try:
                result = self._prelabel_image(image)
                processed += 1

                if result == 'no_dog':
                    no_dog_count += 1

            except Exception as e:
                errors += 1
                if errors <= 10:
                    print(f"\n[PreLabeler] Error processing {image.filename}: {e}")

            # Commit in batches
            if processed % batch_size == 0:
                db.session.commit()
                print(f"\n[PreLabeler] Committed batch ({processed}/{total})")

        # Final commit
        db.session.commit()

        return {
            'processed': processed,
            'errors': errors,
            'total': total,
            'no_dog': no_dog_count
        }

    def _prelabel_image(self, image: Image) -> str:
        """
        Pre-label a single image.

        Returns:
            'labeled', 'no_dog', or 'error'
        """
        # Load image using reconstructed absolute path
        img_path = str(image.absolute_path)
        frame = cv2.imread(img_path)
        if frame is None:
            raise ValueError(f"Could not load image: {img_path}")

        # Detect dogs using backbone
        if self.backbone:
            detections = self.backbone.detect(frame)
        else:
            detections = []

        if not detections:
            # No dog detected
            annotation = CollarAnnotation(
                image_id=image.id,
                model_prediction=CollarLabel.NO_DOG,
                model_confidence=0.0,
                model_p_no_collar=None,
                labeled_by='model'
            )
            image.has_dog = False
            image.status = LabelStatus.PRELABELED
            image.prelabeled_at = datetime.utcnow()

            db.session.add(annotation)
            return 'no_dog'

        # Use first/best detection
        det = detections[0]
        roi = det['roi']
        bbox = det['bbox']

        # Normalize bbox
        h, w = frame.shape[:2]
        norm_bbox = [
            bbox[0] / w, bbox[1] / h,
            bbox[2] / w, bbox[3] / h
        ]

        # Get collar prediction
        if self.collar:
            collar_result = self.collar.predict_with_details(roi)
            p_no_collar = collar_result['p_no_collar']
            confidence = collar_result['confidence']
        else:
            # Fallback heuristic
            p_no_collar = 0.5
            confidence = 0.0

        # Convert to label
        if p_no_collar < 0.5:
            prediction = CollarLabel.WITH_COLLAR
        else:
            prediction = CollarLabel.WITHOUT_COLLAR

        # Create annotation
        annotation = CollarAnnotation(
            image_id=image.id,
            model_prediction=prediction,
            model_confidence=confidence,
            model_p_no_collar=p_no_collar,
            labeled_by='model'
        )

        # Update image
        image.has_dog = True
        image.dog_bbox = str(norm_bbox)
        image.status = LabelStatus.PRELABELED
        image.prelabeled_at = datetime.utcnow()

        db.session.add(annotation)
        return 'labeled'

    def prelabel_single(self, image_id: int) -> dict:
        """
        Pre-label a single image by ID.

        Returns:
            Dict with result details
        """
        image = Image.query.get(image_id)
        if not image:
            return {'error': 'Image not found'}

        try:
            result = self._prelabel_image(image)
            db.session.commit()

            return {
                'success': True,
                'image_id': image_id,
                'result': result,
                'prediction': image.collar_label.model_prediction.name if image.collar_label else None,
                'confidence': image.collar_label.model_confidence if image.collar_label else None
            }

        except Exception as e:
            db.session.rollback()
            return {'error': str(e)}


def prelabel_all_images(batch_size=100, skip_existing=True):
    """
    Convenience function to pre-label all images.

    Args:
        batch_size: Commit after this many images
        skip_existing: Skip images that already have annotations

    Returns:
        Dict with processing statistics
    """
    prelabeler = PreLabeler()
    return prelabeler.prelabel_all(batch_size, skip_existing)
