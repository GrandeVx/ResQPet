"""
Alert Manager
Handles alert generation, storage, and notification
"""

import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from threading import Lock

from app.config import ALERT_COOLDOWN_SECONDS


@dataclass
class Alert:
    """Alert data structure"""
    id: str
    camera_id: str
    timestamp: str
    stray_index: float
    status: str
    bbox: List[float]
    breed: Optional[str]
    thumbnail_path: Optional[str]
    acknowledged: bool = False


class AlertManager:
    """
    Manages stray dog alerts.

    Features:
    - SQLite persistence
    - Cooldown to prevent duplicate alerts
    - Alert acknowledgment
    - Statistics
    """

    def __init__(self, db_path: str = "alerts.db"):
        """
        Initialize alert manager.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self._lock = Lock()
        self._cooldowns: Dict[str, datetime] = {}  # camera_id -> last_alert_time

        self._init_db()

    def _init_db(self):
        """Initialize SQLite database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    stray_index REAL NOT NULL,
                    status TEXT NOT NULL,
                    bbox TEXT NOT NULL,
                    breed TEXT,
                    thumbnail_path TEXT,
                    acknowledged INTEGER DEFAULT 0
                )
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON alerts(timestamp)
            ''')
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_camera ON alerts(camera_id)
            ''')
            conn.commit()

    def should_alert(self, camera_id: str, stray_index: float) -> bool:
        """
        Check if alert should be triggered.

        Args:
            camera_id: Camera identifier
            stray_index: Current stray index

        Returns:
            True if alert should be generated
        """
        # Only alert for likely stray dogs
        if stray_index < 0.7:
            return False

        # Check cooldown
        with self._lock:
            last_alert = self._cooldowns.get(camera_id)
            if last_alert:
                elapsed = (datetime.now() - last_alert).total_seconds()
                if elapsed < ALERT_COOLDOWN_SECONDS:
                    return False

        return True

    def create_alert(self, camera_id: str, detection: Dict,
                    thumbnail_path: Optional[str] = None) -> Optional[Alert]:
        """
        Create a new alert.

        Args:
            camera_id: Camera that detected the dog
            detection: Detection dictionary from StrayIndexCalculator
            thumbnail_path: Path to saved thumbnail image

        Returns:
            Created Alert or None if cooldown active
        """
        stray_index = detection.get('stray_index', 0)

        if not self.should_alert(camera_id, stray_index):
            return None

        # Generate alert ID
        timestamp = datetime.now()
        alert_id = f"{camera_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}"

        alert = Alert(
            id=alert_id,
            camera_id=camera_id,
            timestamp=timestamp.isoformat(),
            stray_index=stray_index,
            status=detection.get('status', 'unknown'),
            bbox=detection.get('bbox', []),
            breed=detection.get('breed'),
            thumbnail_path=thumbnail_path
        )

        # Save to database
        self._save_alert(alert)

        # Update cooldown
        with self._lock:
            self._cooldowns[camera_id] = timestamp

        return alert

    def _save_alert(self, alert: Alert):
        """Save alert to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO alerts (id, camera_id, timestamp, stray_index,
                                   status, bbox, breed, thumbnail_path, acknowledged)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                alert.id,
                alert.camera_id,
                alert.timestamp,
                alert.stray_index,
                alert.status,
                json.dumps(alert.bbox),
                alert.breed,
                alert.thumbnail_path,
                0
            ))
            conn.commit()

    def get_recent_alerts(self, limit: int = 50,
                         camera_id: Optional[str] = None) -> List[Dict]:
        """
        Get recent alerts.

        Args:
            limit: Maximum number of alerts to return
            camera_id: Filter by camera (optional)

        Returns:
            List of alert dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row

            if camera_id:
                cursor = conn.execute('''
                    SELECT * FROM alerts
                    WHERE camera_id = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (camera_id, limit))
            else:
                cursor = conn.execute('''
                    SELECT * FROM alerts
                    ORDER BY timestamp DESC
                    LIMIT ?
                ''', (limit,))

            alerts = []
            for row in cursor.fetchall():
                alerts.append({
                    'id': row['id'],
                    'camera_id': row['camera_id'],
                    'timestamp': row['timestamp'],
                    'stray_index': row['stray_index'],
                    'status': row['status'],
                    'bbox': json.loads(row['bbox']),
                    'breed': row['breed'],
                    'thumbnail_path': row['thumbnail_path'],
                    'acknowledged': bool(row['acknowledged'])
                })

            return alerts

    def acknowledge_alert(self, alert_id: str) -> bool:
        """
        Mark alert as acknowledged.

        Args:
            alert_id: Alert identifier

        Returns:
            True if successful
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE alerts SET acknowledged = 1 WHERE id = ?
                ''', (alert_id,))
                conn.commit()
                return conn.total_changes > 0
        except Exception:
            return False

    def get_statistics(self, hours: int = 24) -> Dict:
        """
        Get alert statistics.

        Args:
            hours: Time window in hours

        Returns:
            Statistics dictionary
        """
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # Total alerts
            total = conn.execute('''
                SELECT COUNT(*) FROM alerts WHERE timestamp > ?
            ''', (cutoff,)).fetchone()[0]

            # Alerts by status
            by_status = {}
            for row in conn.execute('''
                SELECT status, COUNT(*) FROM alerts
                WHERE timestamp > ?
                GROUP BY status
            ''', (cutoff,)):
                by_status[row[0]] = row[1]

            # Alerts by camera
            by_camera = {}
            for row in conn.execute('''
                SELECT camera_id, COUNT(*) FROM alerts
                WHERE timestamp > ?
                GROUP BY camera_id
            ''', (cutoff,)):
                by_camera[row[0]] = row[1]

            # Average stray index
            avg_si = conn.execute('''
                SELECT AVG(stray_index) FROM alerts WHERE timestamp > ?
            ''', (cutoff,)).fetchone()[0] or 0.0

            # Unacknowledged count
            unack = conn.execute('''
                SELECT COUNT(*) FROM alerts
                WHERE timestamp > ? AND acknowledged = 0
            ''', (cutoff,)).fetchone()[0]

            return {
                'total_alerts': total,
                'by_status': by_status,
                'by_camera': by_camera,
                'avg_stray_index': round(avg_si, 3),
                'unacknowledged': unack,
                'time_window_hours': hours
            }

    def clear_old_alerts(self, days: int = 30):
        """
        Remove alerts older than specified days.

        Args:
            days: Age threshold in days
        """
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM alerts WHERE timestamp < ?', (cutoff,))
            conn.commit()
