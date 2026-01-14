"""
WebSocket handlers for real-time CCTV streaming
"""

from flask_socketio import emit, join_room, leave_room
from flask import current_app
from datetime import datetime
import base64
import numpy as np
import cv2

# Global reference to calculator (initialized on first use)
_calculator = None


def get_calculator():
    """Lazy initialization of StrayIndexCalculator"""
    global _calculator
    if _calculator is None:
        try:
            from app.models.fusion import StrayIndexCalculator
            _calculator = StrayIndexCalculator()
            print("[WebSocket] StrayIndexCalculator initialized")
        except Exception as e:
            print(f"[WebSocket] Failed to initialize calculator: {e}")
            _calculator = None
    return _calculator


def register_socketio_handlers(socketio):
    """Register all WebSocket event handlers"""

    @socketio.on('connect')
    def handle_connect():
        """Client connected"""
        print(f"[WebSocket] Client connected: {datetime.now()}")
        emit('connection_status', {'status': 'connected', 'message': 'Welcome to ResQPet'})

    @socketio.on('disconnect')
    def handle_disconnect():
        """Client disconnected"""
        print(f"[WebSocket] Client disconnected: {datetime.now()}")

    @socketio.on('join_camera')
    def handle_join_camera(data):
        """Client subscribes to a camera feed"""
        camera_id = data.get('camera_id')
        if camera_id:
            join_room(f'camera_{camera_id}')
            emit('camera_joined', {
                'camera_id': camera_id,
                'message': f'Subscribed to camera {camera_id}'
            })
            print(f"[WebSocket] Client joined camera: {camera_id}")

    @socketio.on('leave_camera')
    def handle_leave_camera(data):
        """Client unsubscribes from a camera feed"""
        camera_id = data.get('camera_id')
        if camera_id:
            leave_room(f'camera_{camera_id}')
            emit('camera_left', {
                'camera_id': camera_id,
                'message': f'Unsubscribed from camera {camera_id}'
            })

    @socketio.on('analyze_frame')
    def handle_analyze_frame(data):
        """
        Receive a frame from frontend, analyze it, and send back results.

        Expected data:
        {
            'camera_id': str,
            'frame': str (base64 encoded JPEG),
            'timestamp': str (ISO format)
        }
        """
        camera_id = data.get('camera_id', 'unknown')
        frame_b64 = data.get('frame')
        timestamp = data.get('timestamp', datetime.now().isoformat())

        print(f"[WebSocket] Received frame from {camera_id}, size: {len(frame_b64) if frame_b64 else 0} bytes")

        if not frame_b64:
            emit('analysis_error', {'error': 'No frame data provided'})
            return

        try:
            # Decode base64 frame
            frame_bytes = base64.b64decode(frame_b64)
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[WebSocket] Failed to decode frame from {camera_id}")
                emit('analysis_error', {'error': 'Failed to decode frame'})
                return

            print(f"[WebSocket] Frame decoded: {frame.shape}")

            # Get calculator and analyze
            calculator = get_calculator()

            if calculator is None:
                # Fallback: return frame without analysis
                emit('frame_update', {
                    'camera_id': camera_id,
                    'timestamp': timestamp,
                    'frame': frame_b64,
                    'detections': []
                })
                return

            # Analyze frame
            analysis = calculator.analyze_frame(frame)
            detections = analysis.get('detections', [])

            # Only send detection_result if dogs were detected
            if detections:
                print(f"[WebSocket] {len(detections)} dog(s) detected in {camera_id}")

                # Send lightweight detection result (no frame data)
                emit('detection_result', {
                    'camera_id': camera_id,
                    'timestamp': timestamp,
                    'detections': detections
                })

                # Check for alerts: both "possibly_lost" (>=0.3) and "likely_stray" (>=0.7)
                for det in detections:
                    stray_index = det.get('stray_index', 0)
                    # Alert for possibly_lost (>=0.3) and likely_stray (>=0.7)
                    if stray_index >= 0.3:
                        # Draw bounding box on frame for snapshot
                        annotated_frame = frame.copy()
                        bbox = det.get('bbox', [])
                        status = det.get('status', 'unknown')

                        if len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)

                            # Color based on status
                            if status == 'owned':
                                color = (0, 255, 0)  # Green
                            elif status == 'possibly_lost':
                                color = (0, 255, 255)  # Yellow
                            else:
                                color = (0, 0, 255)  # Red

                            # Draw bounding box
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)

                            # Draw label background
                            label = f"SI: {stray_index:.2f} - {status}"
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                            cv2.rectangle(annotated_frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)

                            # Draw label text
                            cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            # Draw breed if available
                            breed = det.get('breed')
                            if breed:
                                cv2.putText(annotated_frame, breed, (x1, y2 + 25),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                        # Encode annotated frame as snapshot
                        _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        snapshot_b64 = base64.b64encode(buffer).decode('utf-8')

                        alert_type = 'STRAY' if stray_index >= 0.7 else 'POSSIBLY LOST'
                        print(f"[WebSocket] ALERT: {alert_type} detected (SI: {stray_index:.2f})")

                        broadcast_alert(socketio, {
                            'camera_id': camera_id,
                            'stray_index': stray_index,
                            'status': status,
                            'breed': det.get('breed'),
                            'timestamp': timestamp,
                            'snapshot': snapshot_b64,
                            'bbox': bbox,
                            'components': det.get('components', {})
                        })

        except Exception as e:
            print(f"[WebSocket] Frame analysis error: {e}")
            import traceback
            traceback.print_exc()
            emit('analysis_error', {'error': str(e)})

    @socketio.on('start_analysis')
    def handle_start_analysis(data):
        """Start real-time analysis for a camera"""
        camera_id = data.get('camera_id')
        print(f"[WebSocket] Starting analysis for camera: {camera_id}")
        emit('analysis_started', {
            'camera_id': camera_id,
            'status': 'processing'
        })

    @socketio.on('stop_analysis')
    def handle_stop_analysis(data):
        """Stop real-time analysis for a camera"""
        camera_id = data.get('camera_id')
        emit('analysis_stopped', {
            'camera_id': camera_id,
            'status': 'stopped'
        })


def broadcast_detection(socketio, camera_id, detection_data):
    """
    Broadcast a detection result to all clients watching a camera

    Args:
        socketio: SocketIO instance
        camera_id: Camera identifier
        detection_data: Detection results including stray_index
    """
    socketio.emit('detection', {
        'camera_id': camera_id,
        'timestamp': datetime.now().isoformat(),
        'data': detection_data
    }, room=f'camera_{camera_id}')


def broadcast_alert(socketio, alert_data):
    """
    Broadcast an alert to all connected clients

    Args:
        socketio: SocketIO instance
        alert_data: Alert information
    """
    socketio.emit('alert', {
        'timestamp': datetime.now().isoformat(),
        'data': alert_data
    })


def broadcast_frame(socketio, camera_id, frame_base64, detections):
    """
    Broadcast a processed frame to clients

    Args:
        socketio: SocketIO instance
        camera_id: Camera identifier
        frame_base64: Base64 encoded frame image
        detections: List of detections with bounding boxes
    """
    socketio.emit('frame_update', {
        'camera_id': camera_id,
        'timestamp': datetime.now().isoformat(),
        'frame': frame_base64,
        'detections': detections
    }, room=f'camera_{camera_id}')
