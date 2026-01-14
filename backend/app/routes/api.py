"""
REST API endpoints for ResQPet
"""

from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import os
import uuid
from pathlib import Path

api_bp = Blueprint('api', __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@api_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ResQPet API',
        'version': '1.0.0'
    })


@api_bp.route('/analyze', methods=['POST'])
def analyze_image():
    """
    Analyze a single image for stray dog detection
    Returns: Stray Index and component probabilities
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    try:
        # Import here to avoid circular imports
        from app.models.fusion import StrayIndexCalculator

        # Save temporarily
        filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
        upload_dir = Path(current_app.root_path) / 'uploads'
        upload_dir.mkdir(exist_ok=True)
        filepath = upload_dir / filename
        file.save(str(filepath))

        # Analyze
        calculator = StrayIndexCalculator()
        results = calculator.analyze_image(str(filepath))

        # Cleanup
        os.remove(filepath)

        return jsonify({
            'success': True,
            'results': results
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@api_bp.route('/cameras', methods=['GET'])
def get_cameras():
    """Get list of configured cameras"""
    # Simulated cameras for demo
    cameras = [
        {'id': 'cam1', 'name': 'Camera 1 - Parco Nord', 'status': 'active'},
        {'id': 'cam2', 'name': 'Camera 2 - Via Roma', 'status': 'active'},
        {'id': 'cam3', 'name': 'Camera 3 - Stazione', 'status': 'active'},
        {'id': 'cam4', 'name': 'Camera 4 - Centro', 'status': 'active'},
    ]
    return jsonify({'cameras': cameras})


@api_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """Get recent alerts"""
    from app.services.alert_manager import AlertManager

    alert_manager = AlertManager()
    alerts = alert_manager.get_recent_alerts(limit=50)

    return jsonify({'alerts': alerts})


@api_bp.route('/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    # Placeholder stats
    stats = {
        'total_detections': 0,
        'total_alerts': 0,
        'active_cameras': 4,
        'avg_stray_index': 0.0,
        'detections_by_category': {
            'owned': 0,
            'possibly_lost': 0,
            'likely_stray': 0
        }
    }
    return jsonify(stats)


# ============================================================
# DEMO ENDPOINTS - Simulated data for frontend demo
# ============================================================

@api_bp.route('/demo/cameras', methods=['GET'])
def get_demo_cameras():
    """Get demo cameras configuration with video URLs and locations"""
    import json

    config_path = Path(current_app.root_path) / 'data' / 'demo_cameras.json'

    if config_path.exists():
        with open(config_path, 'r') as f:
            return jsonify(json.load(f))

    return jsonify({'cameras': [], 'error': 'Demo config not found'}), 404


@api_bp.route('/demo/stats', methods=['GET'])
def get_demo_stats():
    """Get simulated statistics for demo dashboard"""
    import random

    # Generate realistic-looking demo stats
    detections_today = random.randint(35, 85)
    alerts_today = random.randint(5, 15)

    return jsonify({
        'total_detections': random.randint(1200, 2500),
        'active_alerts': random.randint(3, 12),
        'avg_stray_index': round(random.uniform(0.28, 0.52), 2),
        'cameras_online': 9,
        'cameras_total': 9,
        'detections_today': detections_today,
        'alerts_today': alerts_today,
        'system_uptime': '99.7%',
        'last_detection': 'da 2 minuti',
        'history': {
            'labels': ['Lun', 'Mar', 'Mer', 'Gio', 'Ven', 'Sab', 'Dom'],
            'detections': [random.randint(25, 65) for _ in range(7)],
            'alerts': [random.randint(3, 18) for _ in range(7)]
        },
        'categories': {
            'owned': random.randint(60, 75),
            'possibly_lost': random.randint(15, 25),
            'likely_stray': random.randint(5, 15)
        }
    })
