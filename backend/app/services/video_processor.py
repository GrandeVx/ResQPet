"""
Video Processor
Handles video stream processing and frame analysis
"""

import cv2
import base64
import time
import threading
from typing import Dict, Optional, Callable, List
import numpy as np
from pathlib import Path
from queue import Queue

from app.models.fusion import StrayIndexCalculator
from app.services.alert_manager import AlertManager


class VideoProcessor:
    """
    Processes video streams for stray dog detection.

    Supports:
    - Video files (mp4, avi, mov)
    - Simulated CCTV feeds (image sequences)
    - Real-time frame callbacks via WebSocket
    """

    def __init__(self,
                 calculator: Optional[StrayIndexCalculator] = None,
                 alert_manager: Optional[AlertManager] = None,
                 target_fps: int = 10):
        """
        Initialize video processor.

        Args:
            calculator: StrayIndexCalculator instance
            alert_manager: AlertManager instance
            target_fps: Target frames per second for processing
        """
        self.calculator = calculator or StrayIndexCalculator()
        self.alert_manager = alert_manager or AlertManager()
        self.target_fps = target_fps

        self._active_streams: Dict[str, threading.Thread] = {}
        self._stop_flags: Dict[str, threading.Event] = {}

    def process_video_file(self, video_path: str, camera_id: str,
                          frame_callback: Optional[Callable] = None) -> Dict:
        """
        Process a video file.

        Args:
            video_path: Path to video file
            camera_id: Identifier for the camera/stream
            frame_callback: Callback function for each processed frame

        Returns:
            Processing results summary
        """
        if not Path(video_path).exists():
            return {'error': f'Video file not found: {video_path}'}

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Failed to open video: {video_path}'}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps / self.target_fps))

        results = {
            'video_path': video_path,
            'camera_id': camera_id,
            'total_frames': total_frames,
            'processed_frames': 0,
            'total_detections': 0,
            'alerts_generated': 0,
            'detections': []
        }

        frame_idx = 0
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames to match target FPS
                if frame_idx % frame_interval != 0:
                    frame_idx += 1
                    continue

                # Process frame
                analysis = self.calculator.analyze_frame(frame)
                results['processed_frames'] += 1
                results['total_detections'] += analysis['num_detections']

                # Check for alerts
                for detection in analysis['detections']:
                    if detection['stray_index'] >= 0.7:
                        alert = self.alert_manager.create_alert(
                            camera_id, detection
                        )
                        if alert:
                            results['alerts_generated'] += 1

                # Callback with processed frame
                if frame_callback and analysis['detections']:
                    annotated = self._annotate_frame(frame, analysis['detections'])
                    frame_b64 = self._frame_to_base64(annotated)
                    frame_callback(camera_id, frame_b64, analysis['detections'])

                frame_idx += 1

        finally:
            cap.release()

        return results

    def start_stream(self, source: str, camera_id: str,
                    frame_callback: Callable,
                    stop_event: Optional[threading.Event] = None):
        """
        Start processing a video stream in background.

        Args:
            source: Video file path or camera index
            camera_id: Camera identifier
            frame_callback: Callback for processed frames
            stop_event: Event to signal stop
        """
        if camera_id in self._active_streams:
            self.stop_stream(camera_id)

        stop_flag = stop_event or threading.Event()
        self._stop_flags[camera_id] = stop_flag

        thread = threading.Thread(
            target=self._stream_loop,
            args=(source, camera_id, frame_callback, stop_flag),
            daemon=True
        )
        self._active_streams[camera_id] = thread
        thread.start()

    def stop_stream(self, camera_id: str):
        """Stop a running stream"""
        if camera_id in self._stop_flags:
            self._stop_flags[camera_id].set()

        if camera_id in self._active_streams:
            thread = self._active_streams[camera_id]
            thread.join(timeout=2.0)
            del self._active_streams[camera_id]
            del self._stop_flags[camera_id]

    def _stream_loop(self, source: str, camera_id: str,
                    frame_callback: Callable,
                    stop_flag: threading.Event):
        """Internal stream processing loop"""
        # Determine source type
        if source.isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)

        if not cap.isOpened():
            print(f"[VideoProcessor] Failed to open source: {source}")
            return

        frame_time = 1.0 / self.target_fps

        try:
            while not stop_flag.is_set():
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    # Loop video or wait for camera
                    if Path(source).is_file():
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        time.sleep(0.1)
                        continue

                # Process frame
                analysis = self.calculator.analyze_frame(frame)

                # Generate alerts
                for detection in analysis['detections']:
                    if detection['stray_index'] >= 0.7:
                        self.alert_manager.create_alert(camera_id, detection)

                # Send processed frame
                annotated = self._annotate_frame(frame, analysis['detections'])
                frame_b64 = self._frame_to_base64(annotated)
                frame_callback(camera_id, frame_b64, analysis['detections'])

                # Maintain target FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            cap.release()

    def _annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Add detection overlays to frame"""
        output = frame.copy()

        for det in detections:
            bbox = det.get('bbox', [])
            if len(bbox) != 4:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            stray_index = det.get('stray_index', 0)
            status = det.get('status', 'unknown')
            color_hex = det.get('status_color', '#6b7280')

            # Convert hex to BGR
            color = tuple(int(color_hex.lstrip('#')[i:i+2], 16) for i in (4, 2, 0))

            # Draw box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label
            label = f"SI: {stray_index:.2f}"
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return output

    def _frame_to_base64(self, frame: np.ndarray, quality: int = 80) -> str:
        """Convert frame to base64 JPEG"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return base64.b64encode(buffer).decode('utf-8')

    def simulate_cctv_feed(self, images_dir: str, camera_id: str,
                          frame_callback: Callable,
                          loop: bool = True):
        """
        Simulate CCTV feed from image directory.

        Args:
            images_dir: Directory containing images
            camera_id: Camera identifier
            frame_callback: Callback for processed frames
            loop: Whether to loop through images
        """
        images_path = Path(images_dir)
        if not images_path.exists():
            print(f"[VideoProcessor] Images directory not found: {images_dir}")
            return

        # Get image files
        extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        images = sorted([
            f for f in images_path.iterdir()
            if f.suffix.lower() in extensions
        ])

        if not images:
            print(f"[VideoProcessor] No images found in: {images_dir}")
            return

        stop_flag = threading.Event()
        self._stop_flags[camera_id] = stop_flag

        def simulation_loop():
            frame_time = 1.0 / self.target_fps
            idx = 0

            while not stop_flag.is_set():
                start_time = time.time()

                # Load image
                img_path = images[idx % len(images)]
                frame = cv2.imread(str(img_path))

                if frame is not None:
                    # Process
                    analysis = self.calculator.analyze_frame(frame)

                    # Generate alerts
                    for detection in analysis['detections']:
                        if detection['stray_index'] >= 0.7:
                            self.alert_manager.create_alert(camera_id, detection)

                    # Send frame
                    annotated = self._annotate_frame(frame, analysis['detections'])
                    frame_b64 = self._frame_to_base64(annotated)
                    frame_callback(camera_id, frame_b64, analysis['detections'])

                idx += 1
                if not loop and idx >= len(images):
                    break

                # Maintain FPS
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_time - elapsed)
                time.sleep(sleep_time)

        thread = threading.Thread(target=simulation_loop, daemon=True)
        self._active_streams[camera_id] = thread
        thread.start()


# Import numpy for type hints
import numpy as np
