"""
Main labeling UI routes
"""

from flask import Blueprint, render_template, request, redirect, url_for
from labeling_tool.database import db
from labeling_tool.database.models import Image, Dataset, CollarAnnotation, LabelStatus
from labeling_tool.routes.utils import require_user, get_current_user_id

labeling_bp = Blueprint('labeling', __name__)


@labeling_bp.route('/')
@require_user
def index():
    """Home page with quick stats and navigation (user-specific)."""
    user_id = get_current_user_id()
    from labeling_tool.services.statistics import get_user_quick_stats
    stats = get_user_quick_stats(user_id)
    return render_template('index.html', stats=stats)


@labeling_bp.route('/label')
@require_user
def labeling_workspace():
    """Main labeling interface (user-specific images)."""
    dataset_filter = request.args.get('dataset', None)
    confidence_max = request.args.get('confidence_max', 1.0, type=float)

    datasets = Dataset.query.all()

    return render_template('labeling/workspace.html',
                           dataset_filter=dataset_filter,
                           confidence_max=confidence_max,
                           datasets=datasets)


@labeling_bp.route('/review')
@require_user
def review_labels():
    """Review already labeled images (user-specific)."""
    user_id = get_current_user_id()
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 50, type=int)
    label_filter = request.args.get('label', None)

    # Filter by current user
    query = Image.query.filter(
        Image.status == LabelStatus.VERIFIED,
        Image.assigned_user_id == user_id
    )

    if label_filter:
        query = query.join(CollarAnnotation).filter(
            CollarAnnotation.human_label == label_filter
        )

    images = query.order_by(Image.labeled_at.desc()).paginate(
        page=page, per_page=per_page, error_out=False
    )

    return render_template('labeling/review.html', images=images)
