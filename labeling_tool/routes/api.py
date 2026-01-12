"""
REST API routes for labeling operations
"""

from flask import Blueprint, jsonify, request, send_file, current_app
from datetime import datetime
from labeling_tool.database import db
from labeling_tool.database.models import (
    Image, Dataset, CollarAnnotation, LabelStatus, CollarLabel
)
from labeling_tool.routes.utils import require_user, get_current_user_id

api_bp = Blueprint('api', __name__)


@api_bp.route('/image/<int:image_id>')
def serve_image(image_id):
    """Serve image from reconstructed absolute path."""
    image = Image.query.get_or_404(image_id)
    return send_file(str(image.absolute_path))


@api_bp.route('/next')
@require_user
def get_next_image():
    """Get next image to label based on filters (filtered by current user)."""
    user_id = get_current_user_id()
    dataset_name = request.args.get('dataset', None)
    confidence_max = request.args.get('confidence_max', 1.0, type=float)

    # Query prelabeled images for current user only
    query = Image.query.filter(
        Image.status == LabelStatus.PRELABELED,
        Image.assigned_user_id == user_id
    )

    # Filter by dataset
    if dataset_name:
        query = query.join(Dataset).filter(Dataset.name == dataset_name)

    # Order by confidence (low first - harder cases need human review)
    query = query.join(CollarAnnotation)

    if confidence_max < 1.0:
        query = query.filter(CollarAnnotation.model_confidence <= confidence_max)

    query = query.order_by(CollarAnnotation.model_confidence.asc())

    image = query.first()

    if image:
        return jsonify({
            'id': image.id,
            'url': f'/api/image/{image.id}',
            'filename': image.filename,
            'dataset': image.dataset.name,
            'width': image.width,
            'height': image.height,
            'prediction': image.collar_label.model_prediction.name if image.collar_label else None,
            'prediction_value': image.collar_label.model_prediction.value if image.collar_label else None,
            'confidence': image.collar_label.model_confidence if image.collar_label else None,
            'p_no_collar': image.collar_label.model_p_no_collar if image.collar_label else None,
        })
    else:
        return jsonify({'message': 'No more images to label'}), 204


@api_bp.route('/label/<int:image_id>', methods=['POST'])
@require_user
def submit_label(image_id):
    """Submit human label for an image."""
    user_id = get_current_user_id()
    image = Image.query.get_or_404(image_id)

    # Verify image belongs to current user
    if image.assigned_user_id != user_id:
        return jsonify({'error': 'Non autorizzato - immagine assegnata ad altro utente'}), 403

    data = request.json

    label_value = data.get('label')
    notes = data.get('notes', '')

    # Get or create annotation
    annotation = image.collar_label
    if annotation is None:
        annotation = CollarAnnotation(image_id=image_id)
        db.session.add(annotation)

    # Update annotation
    annotation.human_label = CollarLabel(label_value)
    annotation.labeled_by = 'human'
    annotation.labeled_by_user_id = user_id  # Track which user labeled
    annotation.notes = notes
    annotation.updated_at = datetime.utcnow()

    # Check if human corrected the model
    if annotation.model_prediction and annotation.model_prediction != annotation.human_label:
        annotation.human_corrected = True

    # Update image status
    image.status = LabelStatus.VERIFIED
    image.labeled_at = datetime.utcnow()

    db.session.commit()

    return jsonify({
        'success': True,
        'image_id': image_id,
        'label': annotation.human_label.name,
        'corrected': annotation.human_corrected
    })


@api_bp.route('/skip/<int:image_id>', methods=['POST'])
@require_user
def skip_image(image_id):
    """Mark image as skipped."""
    user_id = get_current_user_id()
    image = Image.query.get_or_404(image_id)

    # Verify image belongs to current user
    if image.assigned_user_id != user_id:
        return jsonify({'error': 'Non autorizzato - immagine assegnata ad altro utente'}), 403

    image.status = LabelStatus.SKIPPED
    db.session.commit()
    return jsonify({'success': True, 'image_id': image_id})


@api_bp.route('/stats')
@require_user
def get_stats():
    """Get current labeling statistics for current user."""
    user_id = get_current_user_id()
    from labeling_tool.services.statistics import get_user_statistics
    return jsonify(get_user_statistics(user_id))


@api_bp.route('/datasets')
def get_datasets():
    """Get list of datasets."""
    datasets = Dataset.query.all()
    return jsonify([{
        'id': ds.id,
        'name': ds.name,
        'image_count': ds.image_count
    } for ds in datasets])
