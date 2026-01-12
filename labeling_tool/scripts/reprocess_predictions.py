#!/usr/bin/env python3
"""
Reprocess all collar predictions using the updated model.

This script re-runs the collar detector on ALL images that have dog detections,
updating the model predictions while preserving human labels.

Usage:
    python -m labeling_tool.scripts.reprocess_predictions [--batch-size 100]
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import cv2
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Reprocess collar predictions with updated model'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=100,
        help='Commit after this many images (default: 100)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Process only this many images (for testing)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    args = parser.parse_args()

    # Create Flask app context
    from labeling_tool.app import create_app
    app = create_app()

    with app.app_context():
        from labeling_tool.database import db
        from labeling_tool.database.models import Image, CollarAnnotation, LabelStatus, CollarLabel
        from labeling_tool import config

        print("\n" + "=" * 60)
        print("ResQPet - Reprocess Collar Predictions")
        print("=" * 60)

        # Load models
        print("\n[1/4] Loading models...")

        backend_path = config.PROJECT_ROOT / 'backend'
        sys.path.insert(0, str(backend_path))

        from app.models.backbone import DogPoseBackbone
        from app.models.collar import CollarDetector

        backbone = DogPoseBackbone(
            model_path=str(config.BACKBONE_MODEL),
            confidence=0.5
        )
        print(f"  Backbone: {config.BACKBONE_MODEL.name}")

        collar = CollarDetector(
            model_path=str(config.COLLAR_MODEL),
            confidence=0.5
        )
        print(f"  Collar: {config.COLLAR_MODEL.name}")

        # Query images with dog detections
        print("\n[2/4] Finding images to reprocess...")
        query = Image.query.filter(Image.has_dog == True)

        if args.limit:
            query = query.limit(args.limit)

        total = query.count()
        print(f"  Found {total} images with dogs")

        if total == 0:
            print("\nNo images to process!")
            return 0

        if args.dry_run:
            print("\n[DRY RUN] Would process these images:")
            for img in query.limit(10):
                print(f"  - {img.filename}")
            if total > 10:
                print(f"  ... and {total - 10} more")
            return 0

        # Process images
        print("\n[3/4] Reprocessing predictions...")
        processed = 0
        updated = 0
        errors = 0
        stats = {'improved': 0, 'same': 0, 'changed_class': 0}

        # Human vs Model comparison
        human_comparison = {
            'total_with_human': 0,
            'matches': 0,
            'mismatches': 0,
            'confusion': {
                'tp_collar': 0,     # Human=WITH, Model=WITH
                'tn_collar': 0,     # Human=WITHOUT, Model=WITHOUT
                'fp_collar': 0,     # Human=WITHOUT, Model=WITH
                'fn_collar': 0,     # Human=WITH, Model=WITHOUT
            }
        }

        for image in tqdm(query.yield_per(args.batch_size), total=total, desc="Processing"):
            try:
                # Load image
                img_path = str(image.absolute_path)
                frame = cv2.imread(img_path)

                if frame is None:
                    errors += 1
                    continue

                # Detect dog
                detections = backbone.detect(frame)

                if not detections:
                    # No dog detected with new model
                    processed += 1
                    continue

                # Get collar prediction
                det = detections[0]
                roi = det['roi']
                bbox = det['bbox']

                collar_result = collar.predict_with_details(roi)
                new_p_no_collar = collar_result['p_no_collar']
                new_confidence = collar_result['confidence']

                # Get or create annotation
                annotation = image.collar_label
                if annotation:
                    old_p = annotation.model_p_no_collar
                    old_conf = annotation.model_confidence
                    old_pred = annotation.model_prediction

                    # Update model predictions (preserve human label)
                    annotation.model_p_no_collar = new_p_no_collar
                    annotation.model_confidence = new_confidence

                    # Update model prediction
                    if new_p_no_collar < 0.5:
                        annotation.model_prediction = CollarLabel.WITH_COLLAR
                    else:
                        annotation.model_prediction = CollarLabel.WITHOUT_COLLAR

                    # Track changes
                    if old_pred != annotation.model_prediction:
                        stats['changed_class'] += 1
                    elif old_p is not None and abs(new_confidence - (old_conf or 0)) > 0.1:
                        stats['improved'] += 1
                    else:
                        stats['same'] += 1

                    # Compare with human label if exists
                    if annotation.human_label is not None:
                        human_comparison['total_with_human'] += 1
                        human = annotation.human_label
                        model = annotation.model_prediction

                        if human == model:
                            human_comparison['matches'] += 1
                        else:
                            human_comparison['mismatches'] += 1

                        # Confusion matrix (for WITH_COLLAR detection)
                        if human == CollarLabel.WITH_COLLAR and model == CollarLabel.WITH_COLLAR:
                            human_comparison['confusion']['tp_collar'] += 1
                        elif human == CollarLabel.WITHOUT_COLLAR and model == CollarLabel.WITHOUT_COLLAR:
                            human_comparison['confusion']['tn_collar'] += 1
                        elif human == CollarLabel.WITHOUT_COLLAR and model == CollarLabel.WITH_COLLAR:
                            human_comparison['confusion']['fp_collar'] += 1
                        elif human == CollarLabel.WITH_COLLAR and model == CollarLabel.WITHOUT_COLLAR:
                            human_comparison['confusion']['fn_collar'] += 1

                    updated += 1
                else:
                    # Create new annotation
                    if new_p_no_collar < 0.5:
                        prediction = CollarLabel.WITH_COLLAR
                    else:
                        prediction = CollarLabel.WITHOUT_COLLAR

                    annotation = CollarAnnotation(
                        image_id=image.id,
                        model_prediction=prediction,
                        model_confidence=new_confidence,
                        model_p_no_collar=new_p_no_collar,
                        labeled_by='model'
                    )
                    db.session.add(annotation)
                    updated += 1

                # Update bbox
                h, w = frame.shape[:2]
                norm_bbox = [
                    bbox[0] / w, bbox[1] / h,
                    bbox[2] / w, bbox[3] / h
                ]
                image.dog_bbox = str(norm_bbox)

                processed += 1

            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"\n  Error: {image.filename}: {e}")

            # Commit in batches
            if processed % args.batch_size == 0:
                db.session.commit()

        # Final commit
        db.session.commit()

        # Summary
        print("\n" + "=" * 60)
        print("[4/4] Reprocessing Complete!")
        print("=" * 60)
        print(f"\n  Total processed: {processed}")
        print(f"  Updated: {updated}")
        print(f"  Errors: {errors}")
        print(f"\n  Prediction Changes:")
        print(f"    - Class changed: {stats['changed_class']}")
        print(f"    - Confidence improved: {stats['improved']}")
        print(f"    - Same prediction: {stats['same']}")

        # Human vs Model comparison
        if human_comparison['total_with_human'] > 0:
            total_human = human_comparison['total_with_human']
            matches = human_comparison['matches']
            mismatches = human_comparison['mismatches']
            accuracy = matches / total_human * 100

            cm = human_comparison['confusion']
            tp = cm['tp_collar']
            tn = cm['tn_collar']
            fp = cm['fp_collar']
            fn = cm['fn_collar']

            # Calculate metrics
            precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            print("\n" + "-" * 60)
            print("  CONFRONTO MODELLO vs ETICHETTE UMANE")
            print("-" * 60)
            print(f"\n  Immagini con etichetta umana: {total_human}")
            print(f"\n  ACCURACY: {accuracy:.1f}% ({matches}/{total_human})")
            print(f"    - Matches: {matches}")
            print(f"    - Mismatches: {mismatches}")

            print(f"\n  CONFUSION MATRIX (WITH_COLLAR):")
            print(f"                      Predicted")
            print(f"                   WITH    WITHOUT")
            print(f"    Actual WITH    {tp:5d}    {fn:5d}")
            print(f"    Actual WITHOUT {fp:5d}    {tn:5d}")

            print(f"\n  METRICHE (rilevamento WITH_COLLAR):")
            print(f"    - Precision: {precision:.1f}%")
            print(f"    - Recall:    {recall:.1f}%")
            print(f"    - F1 Score:  {f1:.1f}%")
        else:
            print("\n  (Nessuna etichetta umana trovata per confronto)")

        print("\n" + "=" * 60)
        print("Human labels preserved. Model predictions updated.")
        print("=" * 60)

    return 0


if __name__ == '__main__':
    sys.exit(main())
