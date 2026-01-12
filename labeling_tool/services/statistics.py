"""
Statistics calculation service
"""

from labeling_tool.database import db
from labeling_tool.database.models import (
    Image, Dataset, CollarAnnotation, LabelStatus, CollarLabel
)


def get_quick_stats():
    """Get quick overview statistics for home page."""
    total = Image.query.count()
    unlabeled = Image.query.filter(Image.status == LabelStatus.UNLABELED).count()
    prelabeled = Image.query.filter(Image.status == LabelStatus.PRELABELED).count()
    verified = Image.query.filter(Image.status == LabelStatus.VERIFIED).count()
    skipped = Image.query.filter(Image.status == LabelStatus.SKIPPED).count()

    progress_pct = (verified / total * 100) if total > 0 else 0

    return {
        'total': total,
        'unlabeled': unlabeled,
        'prelabeled': prelabeled,
        'verified': verified,
        'skipped': skipped,
        'progress_pct': progress_pct
    }


def get_full_statistics():
    """Get full statistics for dashboard."""
    stats = get_quick_stats()

    # Label distribution (verified only)
    with_collar = CollarAnnotation.query.join(Image).filter(
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.WITH_COLLAR
    ).count()

    without_collar = CollarAnnotation.query.join(Image).filter(
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.WITHOUT_COLLAR
    ).count()

    unclear = CollarAnnotation.query.join(Image).filter(
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.UNCLEAR
    ).count()

    no_dog = CollarAnnotation.query.join(Image).filter(
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.NO_DOG
    ).count()

    # Correction rate
    corrected = CollarAnnotation.query.filter(
        CollarAnnotation.human_corrected == True
    ).count()

    correction_rate = (corrected / stats['verified'] * 100) if stats['verified'] > 0 else 0

    # By dataset
    by_dataset = []
    for ds in Dataset.query.all():
        ds_total = Image.query.filter(Image.dataset_id == ds.id).count()
        ds_prelabeled = Image.query.filter(
            Image.dataset_id == ds.id,
            Image.status == LabelStatus.PRELABELED
        ).count()
        ds_verified = Image.query.filter(
            Image.dataset_id == ds.id,
            Image.status == LabelStatus.VERIFIED
        ).count()
        ds_progress = (ds_verified / ds_total * 100) if ds_total > 0 else 0

        by_dataset.append({
            'name': ds.name,
            'total': ds_total,
            'prelabeled': ds_prelabeled,
            'verified': ds_verified,
            'progress_pct': ds_progress
        })

    # Confidence histogram
    confidence_histogram = get_confidence_histogram()

    stats.update({
        'with_collar': with_collar,
        'without_collar': without_collar,
        'unclear': unclear,
        'no_dog': no_dog,
        'corrected': corrected,
        'correction_rate': correction_rate,
        'by_dataset': by_dataset,
        'confidence_histogram': confidence_histogram
    })

    return stats


def get_confidence_histogram():
    """Get histogram of model confidence values (10 bins)."""
    from sqlalchemy import func

    histogram = [0] * 10

    # Query confidence values
    confidences = db.session.query(CollarAnnotation.model_confidence).filter(
        CollarAnnotation.model_confidence.isnot(None)
    ).all()

    for (conf,) in confidences:
        if conf is not None:
            bin_idx = min(int(conf * 10), 9)
            histogram[bin_idx] += 1

    return histogram


# =============================================================================
# User-specific statistics functions
# =============================================================================

def get_user_quick_stats(user_id):
    """Get quick overview statistics for a specific user."""
    total = Image.query.filter(Image.assigned_user_id == user_id).count()
    unlabeled = Image.query.filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.UNLABELED
    ).count()
    prelabeled = Image.query.filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.PRELABELED
    ).count()
    verified = Image.query.filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.VERIFIED
    ).count()
    skipped = Image.query.filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.SKIPPED
    ).count()

    progress_pct = (verified / total * 100) if total > 0 else 0

    return {
        'total': total,
        'unlabeled': unlabeled,
        'prelabeled': prelabeled,
        'verified': verified,
        'skipped': skipped,
        'progress_pct': progress_pct,
        'user_id': user_id
    }


def get_user_statistics(user_id):
    """Get statistics for API calls (user-specific)."""
    return get_user_quick_stats(user_id)


def get_user_full_statistics(user_id):
    """Get full statistics for dashboard (user-specific)."""
    stats = get_user_quick_stats(user_id)

    # Label distribution (verified only, for this user)
    with_collar = CollarAnnotation.query.join(Image).filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.WITH_COLLAR
    ).count()

    without_collar = CollarAnnotation.query.join(Image).filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.WITHOUT_COLLAR
    ).count()

    unclear = CollarAnnotation.query.join(Image).filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.UNCLEAR
    ).count()

    no_dog = CollarAnnotation.query.join(Image).filter(
        Image.assigned_user_id == user_id,
        Image.status == LabelStatus.VERIFIED,
        CollarAnnotation.human_label == CollarLabel.NO_DOG
    ).count()

    # Correction rate for this user
    corrected = CollarAnnotation.query.join(Image).filter(
        Image.assigned_user_id == user_id,
        CollarAnnotation.human_corrected == True
    ).count()

    correction_rate = (corrected / stats['verified'] * 100) if stats['verified'] > 0 else 0

    # By dataset (for this user)
    by_dataset = []
    for ds in Dataset.query.all():
        ds_total = Image.query.filter(
            Image.dataset_id == ds.id,
            Image.assigned_user_id == user_id
        ).count()
        if ds_total == 0:
            continue
        ds_prelabeled = Image.query.filter(
            Image.dataset_id == ds.id,
            Image.assigned_user_id == user_id,
            Image.status == LabelStatus.PRELABELED
        ).count()
        ds_verified = Image.query.filter(
            Image.dataset_id == ds.id,
            Image.assigned_user_id == user_id,
            Image.status == LabelStatus.VERIFIED
        ).count()
        ds_progress = (ds_verified / ds_total * 100) if ds_total > 0 else 0

        by_dataset.append({
            'name': ds.name,
            'total': ds_total,
            'prelabeled': ds_prelabeled,
            'verified': ds_verified,
            'progress_pct': ds_progress
        })

    # Confidence histogram for this user
    confidence_histogram = get_user_confidence_histogram(user_id)

    stats.update({
        'with_collar': with_collar,
        'without_collar': without_collar,
        'unclear': unclear,
        'no_dog': no_dog,
        'corrected': corrected,
        'correction_rate': correction_rate,
        'by_dataset': by_dataset,
        'confidence_histogram': confidence_histogram
    })

    return stats


def get_user_confidence_histogram(user_id):
    """Get histogram of model confidence values for a specific user (10 bins)."""
    histogram = [0] * 10

    # Query confidence values for this user's images
    confidences = db.session.query(CollarAnnotation.model_confidence).join(Image).filter(
        Image.assigned_user_id == user_id,
        CollarAnnotation.model_confidence.isnot(None)
    ).all()

    for (conf,) in confidences:
        if conf is not None:
            bin_idx = min(int(conf * 10), 9)
            histogram[bin_idx] += 1

    return histogram
