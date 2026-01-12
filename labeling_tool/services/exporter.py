"""
YOLO format exporter service
"""

import os
import random
import yaml
from pathlib import Path
from datetime import datetime

from labeling_tool.database import db
from labeling_tool.database.models import (
    Image, CollarAnnotation, LabelStatus, CollarLabel, ExportHistory
)


class YOLOExporter:
    """Export verified labels to YOLO format."""

    def __init__(self, export_base_path: str):
        self.export_base = Path(export_base_path)
        self.export_base.mkdir(parents=True, exist_ok=True)

    def export(self, train_split=0.8, min_confidence=0.0,
               include_model_only=False, version=None):
        """
        Export labels to YOLO format.

        Args:
            train_split: Fraction for training set
            min_confidence: Minimum confidence for model-only labels
            include_model_only: Include prelabeled (not human verified) images
            version: Export version number (auto-increment if None)
        """
        # Determine version
        if version is None:
            last_export = ExportHistory.query.order_by(
                ExportHistory.version.desc()
            ).first()
            version = (last_export.version + 1) if last_export else 1

        # Create export directory
        export_dir = self.export_base / f'collar_labels_v{version}'
        export_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories
        (export_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
        (export_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
        (export_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
        (export_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

        # Query images with valid labels
        query = Image.query.join(CollarAnnotation)

        if include_model_only:
            query = query.filter(
                Image.status.in_([LabelStatus.VERIFIED, LabelStatus.PRELABELED])
            )
            if min_confidence > 0:
                query = query.filter(
                    CollarAnnotation.model_confidence >= min_confidence
                )
        else:
            query = query.filter(Image.status == LabelStatus.VERIFIED)

        images = query.all()

        # Filter to valid collar labels
        valid_labels = [CollarLabel.WITH_COLLAR, CollarLabel.WITHOUT_COLLAR]
        valid_images = []
        for img in images:
            if img.collar_label:
                label = img.collar_label.final_label
                if label in valid_labels:
                    valid_images.append(img)

        # Shuffle and split
        random.shuffle(valid_images)
        split_idx = int(len(valid_images) * train_split)
        train_images = valid_images[:split_idx]
        val_images = valid_images[split_idx:]

        # Export images
        train_count = self._export_split(train_images, export_dir, 'train')
        val_count = self._export_split(val_images, export_dir, 'val')

        # Create data.yaml
        self._create_yaml(export_dir)

        # Record export
        export_record = ExportHistory(
            version=version,
            export_path=str(export_dir),
            total_images=train_count + val_count,
            train_images=train_count,
            val_images=val_count
        )
        db.session.add(export_record)
        db.session.commit()

        return {
            'version': version,
            'path': str(export_dir),
            'train_images': train_count,
            'val_images': val_count,
            'total': train_count + val_count
        }

    def _export_split(self, images, export_dir: Path, split: str) -> int:
        """Export a split (train/val)."""
        count = 0

        for img in images:
            try:
                # Create symlink to image
                img_link = export_dir / 'images' / split / img.filename

                # Handle duplicate filenames
                if img_link.exists():
                    stem = img_link.stem
                    suffix = img_link.suffix
                    counter = 1
                    while img_link.exists():
                        img_link = export_dir / 'images' / split / f"{stem}_{counter}{suffix}"
                        counter += 1

                os.symlink(str(img.absolute_path), img_link)

                # Create label file
                label_file = export_dir / 'labels' / split / (
                    img_link.stem + '.txt'
                )

                label = img.collar_label.final_label
                class_id = label.value  # 0 or 1

                # Parse bbox if available
                if img.dog_bbox:
                    import ast
                    bbox = ast.literal_eval(img.dog_bbox)
                    x1, y1, x2, y2 = bbox

                    # Convert to YOLO format: cx, cy, w, h
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1

                    with open(label_file, 'w') as f:
                        f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                else:
                    # No bbox, use full image
                    with open(label_file, 'w') as f:
                        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

                count += 1

            except Exception as e:
                print(f"[Exporter] Error exporting {img.filename}: {e}")

        return count

    def _create_yaml(self, export_dir: Path):
        """Create data.yaml for YOLO training."""
        data = {
            'path': str(export_dir),
            'train': 'images/train',
            'val': 'images/val',
            'nc': 2,
            'names': ['Dog-with-Leash', 'Dog-without-Leash']
        }

        yaml_path = export_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
