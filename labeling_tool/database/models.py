"""
SQLAlchemy ORM models for the labeling tool
"""

from datetime import datetime
import enum
from pathlib import Path
from labeling_tool.database import db


class LabelStatus(enum.Enum):
    """Image labeling status"""
    UNLABELED = "unlabeled"
    PRELABELED = "prelabeled"
    VERIFIED = "verified"
    SKIPPED = "skipped"


class CollarLabel(enum.Enum):
    """Collar classification labels"""
    WITH_COLLAR = 0       # Dog-with-Leash (class 0)
    WITHOUT_COLLAR = 1    # Dog-without-Leash (class 1)
    UNCLEAR = -1          # Cannot determine
    NO_DOG = -2           # No dog in image


class Dataset(db.Model):
    """Represents a source dataset."""
    __tablename__ = 'datasets'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    base_path = db.Column(db.String(500), nullable=False)
    image_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    images = db.relationship("Image", back_populates="dataset", lazy='dynamic')

    @property
    def absolute_base_path(self):
        """Reconstruct absolute path at runtime from relative base_path."""
        from labeling_tool import config
        base = Path(self.base_path)
        # Se già assoluto (legacy), ritorna direttamente
        if base.is_absolute():
            return base
        # Altrimenti combina con PROJECT_ROOT
        return config.PROJECT_ROOT / self.base_path

    def __repr__(self):
        return f"<Dataset {self.name}: {self.image_count} images>"


class Image(db.Model):
    """Unified image index across all datasets."""
    __tablename__ = 'images'

    id = db.Column(db.Integer, primary_key=True)
    dataset_id = db.Column(db.Integer, db.ForeignKey('datasets.id'), nullable=False)

    # File reference (no duplication)
    original_path = db.Column(db.String(500), nullable=False)
    relative_path = db.Column(db.String(300), nullable=False)
    filename = db.Column(db.String(200), nullable=False)

    # Image metadata
    width = db.Column(db.Integer)
    height = db.Column(db.Integer)
    file_size = db.Column(db.Integer)  # bytes

    # Processing status
    status = db.Column(db.Enum(LabelStatus), default=LabelStatus.UNLABELED)
    has_dog = db.Column(db.Boolean)
    dog_bbox = db.Column(db.String(100))  # JSON: [x1, y1, x2, y2] normalized

    # User assignment (1-5)
    assigned_user_id = db.Column(db.Integer, nullable=True, index=True)

    # Timestamps
    indexed_at = db.Column(db.DateTime, default=datetime.utcnow)
    prelabeled_at = db.Column(db.DateTime)
    labeled_at = db.Column(db.DateTime)

    # Relationships
    dataset = db.relationship("Dataset", back_populates="images")
    collar_label = db.relationship("CollarAnnotation", back_populates="image", uselist=False)

    @property
    def absolute_path(self):
        """Reconstruct absolute path at runtime."""
        from labeling_tool import config
        # Se original_path è assoluto e il file esiste, usalo (legacy/retrocompatibilità)
        orig = Path(self.original_path)
        if orig.is_absolute() and orig.exists():
            return orig
        # Altrimenti ricostruisci da dataset.base_path + relative_path
        return self.dataset.absolute_base_path / self.relative_path

    def __repr__(self):
        return f"<Image {self.id}: {self.filename} ({self.status.value})>"


class CollarAnnotation(db.Model):
    """Collar label for an image."""
    __tablename__ = 'collar_annotations'

    id = db.Column(db.Integer, primary_key=True)
    image_id = db.Column(db.Integer, db.ForeignKey('images.id'), unique=True, nullable=False)

    # Model prediction (pre-labeling)
    model_prediction = db.Column(db.Enum(CollarLabel))
    model_confidence = db.Column(db.Float)
    model_p_no_collar = db.Column(db.Float)

    # Human annotation
    human_label = db.Column(db.Enum(CollarLabel))
    human_corrected = db.Column(db.Boolean, default=False)

    # Bounding box for collar region (optional)
    collar_bbox = db.Column(db.String(100))

    # Metadata
    labeled_by = db.Column(db.String(50), default='model')
    labeled_by_user_id = db.Column(db.Integer, nullable=True)  # User ID (1-5) who labeled
    notes = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)

    # Relationship
    image = db.relationship("Image", back_populates="collar_label")

    @property
    def final_label(self):
        """Return human label if available, otherwise model prediction."""
        return self.human_label if self.human_label is not None else self.model_prediction

    def __repr__(self):
        label = self.final_label.name if self.final_label else 'None'
        return f"<CollarAnnotation {self.image_id}: {label}>"


class LabelingSession(db.Model):
    """Track labeling sessions for statistics."""
    __tablename__ = 'labeling_sessions'

    id = db.Column(db.Integer, primary_key=True)
    started_at = db.Column(db.DateTime, default=datetime.utcnow)
    ended_at = db.Column(db.DateTime)
    images_labeled = db.Column(db.Integer, default=0)
    images_corrected = db.Column(db.Integer, default=0)
    images_skipped = db.Column(db.Integer, default=0)


class ExportHistory(db.Model):
    """Track YOLO exports."""
    __tablename__ = 'export_history'

    id = db.Column(db.Integer, primary_key=True)
    version = db.Column(db.Integer, nullable=False)
    export_path = db.Column(db.String(500), nullable=False)
    total_images = db.Column(db.Integer)
    train_images = db.Column(db.Integer)
    val_images = db.Column(db.Integer)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    notes = db.Column(db.Text)
    # Multi-user support
    user_id = db.Column(db.Integer, nullable=True)  # User who triggered export (NULL = all/merge)
    export_type = db.Column(db.String(20), default='yolo')  # 'json', 'yolo', 'merge'
