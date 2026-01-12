"""
Dashboard and statistics routes
"""

from flask import Blueprint, render_template, request, jsonify
from labeling_tool.database.models import ExportHistory
from labeling_tool.routes.utils import require_user, get_current_user_id

dashboard_bp = Blueprint('dashboard', __name__)


@dashboard_bp.route('/')
@require_user
def statistics():
    """Statistics dashboard (user-specific)."""
    user_id = get_current_user_id()
    from labeling_tool.services.statistics import get_user_full_statistics
    stats = get_user_full_statistics(user_id)

    # Get export history for current user
    exports = ExportHistory.query.filter(
        ExportHistory.user_id == user_id
    ).order_by(ExportHistory.version.desc()).limit(10).all()

    return render_template('dashboard/statistics.html', stats=stats, exports=exports)


@dashboard_bp.route('/export', methods=['POST'])
@require_user
def trigger_export():
    """Trigger YOLO format export."""
    from labeling_tool.services.exporter import YOLOExporter
    from labeling_tool import config

    data = request.json or {}
    train_split = data.get('train_split', 0.8)
    min_confidence = data.get('min_confidence', 0.0)
    include_model_only = data.get('include_model_only', False)

    exporter = YOLOExporter(str(config.EXPORTS_DIR))
    result = exporter.export(
        train_split=train_split,
        min_confidence=min_confidence,
        include_model_only=include_model_only
    )

    return jsonify(result)


@dashboard_bp.route('/export-json', methods=['POST'])
@require_user
def trigger_json_export():
    """Trigger JSON export for current user."""
    from labeling_tool.services.json_exporter import JSONExporter
    from labeling_tool import config

    user_id = get_current_user_id()
    data = request.json or {}
    include_unverified = data.get('include_unverified', False)

    exporter = JSONExporter(str(config.EXPORTS_DIR / 'json'))
    result = exporter.export_user(user_id, include_unverified)

    return jsonify(result)
