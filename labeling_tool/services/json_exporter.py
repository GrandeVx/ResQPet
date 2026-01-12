"""
JSON format exporter service for individual users
"""

import json
from pathlib import Path
from datetime import datetime

from labeling_tool.database import db
from labeling_tool.database.models import (
    Image, CollarAnnotation, LabelStatus, CollarLabel, ExportHistory
)
from labeling_tool import config


class JSONExporter:
    """Export user annotations to JSON format."""

    def __init__(self, export_base_path: str):
        self.export_base = Path(export_base_path)
        self.export_base.mkdir(parents=True, exist_ok=True)

    def export_user(self, user_id: int, include_unverified: bool = False) -> dict:
        """
        Export annotations for a specific user to JSON.

        Args:
            user_id: User ID (1-5)
            include_unverified: Include prelabeled images (not human verified)

        Returns:
            dict with export info
        """
        user_name = config.USERS.get(user_id, f"User {user_id}")

        # Query images for this user
        query = Image.query.filter(Image.assigned_user_id == user_id)

        if include_unverified:
            query = query.filter(
                Image.status.in_([LabelStatus.VERIFIED, LabelStatus.PRELABELED])
            )
        else:
            query = query.filter(Image.status == LabelStatus.VERIFIED)

        images = query.all()

        # Build annotations list
        annotations = []
        for img in images:
            ann = img.collar_label
            if not ann:
                continue

            final_label = ann.final_label
            if final_label is None:
                continue

            # Costruisci path relativo completo (dataset_base_path/relative_path)
            dataset_base = img.dataset.base_path if img.dataset else ""
            full_relative_path = f"{dataset_base}/{img.relative_path}" if dataset_base else img.relative_path

            annotations.append({
                "image_id": img.id,
                "image_path": full_relative_path,  # Path relativo a PROJECT_ROOT
                "dataset_base_path": dataset_base,
                "relative_path": img.relative_path,
                "filename": img.filename,
                "dataset": img.dataset.name if img.dataset else None,

                "label": final_label.name,
                "label_value": final_label.value,

                "model_prediction": ann.model_prediction.name if ann.model_prediction else None,
                "model_confidence": ann.model_confidence,
                "human_corrected": ann.human_corrected or False,

                "has_dog": img.has_dog,
                "dog_bbox": json.loads(img.dog_bbox) if img.dog_bbox else None,
                "collar_bbox": json.loads(ann.collar_bbox) if ann.collar_bbox else None,

                "status": img.status.value,
                "labeled_at": img.labeled_at.isoformat() if img.labeled_at else None,
                "notes": ann.notes or ""
            })

        # Determine version
        last_export = ExportHistory.query.filter(
            ExportHistory.user_id == user_id,
            ExportHistory.export_type == 'json'
        ).order_by(ExportHistory.version.desc()).first()

        version = (last_export.version + 1) if last_export else 1

        # Build export data
        export_data = {
            "export_metadata": {
                "user_id": user_id,
                "user_name": user_name,
                "export_date": datetime.utcnow().isoformat() + "Z",
                "total_images": len(annotations),
                "version": f"v{version}"
            },
            "annotations": annotations
        }

        # Save to file
        export_filename = f"user_{user_id}_annotations_v{version}.json"
        export_path = self.export_base / export_filename

        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        # Record export
        export_record = ExportHistory(
            version=version,
            export_path=str(export_path),
            total_images=len(annotations),
            train_images=0,
            val_images=0,
            user_id=user_id,
            export_type='json'
        )
        db.session.add(export_record)
        db.session.commit()

        return {
            "success": True,
            "user_id": user_id,
            "user_name": user_name,
            "version": version,
            "total_images": len(annotations),
            "path": str(export_path),
            "filename": export_filename
        }
