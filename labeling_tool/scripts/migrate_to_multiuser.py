#!/usr/bin/env python3
"""
Migration script to add multi-user support.

This script:
1. Adds new columns to the database (if not exist)
2. Assigns existing VERIFIED images to User 1
3. Distributes PRELABELED/UNLABELED images equally among all 5 users

Usage:
    python -m labeling_tool.scripts.migrate_to_multiuser
"""

import sys
import random
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from labeling_tool.app import create_app
from labeling_tool.database import db
from labeling_tool.database.models import Image, CollarAnnotation, LabelStatus
from labeling_tool import config


def check_column_exists(table_name: str, column_name: str) -> bool:
    """Check if a column exists in a table (SQLite specific)."""
    result = db.session.execute(
        db.text(f"PRAGMA table_info({table_name})")
    ).fetchall()
    columns = [row[1] for row in result]
    return column_name in columns


def add_columns_if_needed():
    """Add new columns to existing tables if they don't exist."""
    print("\n[Step 0] Checking database schema...")

    # Check and add assigned_user_id to images
    if not check_column_exists('images', 'assigned_user_id'):
        print("  Adding 'assigned_user_id' column to images table...")
        db.session.execute(db.text(
            "ALTER TABLE images ADD COLUMN assigned_user_id INTEGER"
        ))
        db.session.execute(db.text(
            "CREATE INDEX IF NOT EXISTS ix_images_assigned_user_id ON images (assigned_user_id)"
        ))
        db.session.commit()
        print("  -> Done")
    else:
        print("  'assigned_user_id' column already exists in images table")

    # Check and add labeled_by_user_id to collar_annotations
    if not check_column_exists('collar_annotations', 'labeled_by_user_id'):
        print("  Adding 'labeled_by_user_id' column to collar_annotations table...")
        db.session.execute(db.text(
            "ALTER TABLE collar_annotations ADD COLUMN labeled_by_user_id INTEGER"
        ))
        db.session.commit()
        print("  -> Done")
    else:
        print("  'labeled_by_user_id' column already exists in collar_annotations table")

    # Check and add user_id to export_history
    if not check_column_exists('export_history', 'user_id'):
        print("  Adding 'user_id' column to export_history table...")
        db.session.execute(db.text(
            "ALTER TABLE export_history ADD COLUMN user_id INTEGER"
        ))
        db.session.commit()
        print("  -> Done")
    else:
        print("  'user_id' column already exists in export_history table")

    # Check and add export_type to export_history
    if not check_column_exists('export_history', 'export_type'):
        print("  Adding 'export_type' column to export_history table...")
        db.session.execute(db.text(
            "ALTER TABLE export_history ADD COLUMN export_type VARCHAR(20) DEFAULT 'yolo'"
        ))
        db.session.commit()
        print("  -> Done")
    else:
        print("  'export_type' column already exists in export_history table")


def migrate():
    """Run the migration."""
    app = create_app()

    with app.app_context():
        print("=" * 60)
        print("ResQPet Multi-User Migration")
        print("=" * 60)

        # Step 0: Add columns if needed
        add_columns_if_needed()

        # Step 1: Assign VERIFIED images to User 1
        print("\n[Step 1] Assigning VERIFIED images to Utente 1...")
        verified_images = Image.query.filter(
            Image.status == LabelStatus.VERIFIED,
            Image.assigned_user_id.is_(None)
        ).all()

        print(f"  Found {len(verified_images)} VERIFIED images without user assignment")

        for img in verified_images:
            img.assigned_user_id = 1

        # Also update their annotations
        for img in verified_images:
            if img.collar_label and img.collar_label.labeled_by_user_id is None:
                img.collar_label.labeled_by_user_id = 1

        db.session.commit()
        print(f"  -> Assigned {len(verified_images)} images to Utente 1")

        # Step 2: Distribute unassigned images equally
        print("\n[Step 2] Distributing unassigned images among 5 users...")
        unassigned = Image.query.filter(
            Image.assigned_user_id.is_(None)
        ).all()

        print(f"  Found {len(unassigned)} unassigned images")

        # Shuffle for random distribution
        random.shuffle(unassigned)

        # Round-robin assignment
        for i, img in enumerate(unassigned):
            user_id = (i % config.NUM_USERS) + 1  # 1, 2, 3, 4, 5, 1, 2, ...
            img.assigned_user_id = user_id

        db.session.commit()
        print(f"  -> Distributed {len(unassigned)} images")

        # Step 3: Print final distribution
        print("\n[Step 3] Final distribution:")
        print("-" * 50)

        total_all = 0
        for user_id in range(1, config.NUM_USERS + 1):
            total = Image.query.filter(Image.assigned_user_id == user_id).count()
            verified = Image.query.filter(
                Image.assigned_user_id == user_id,
                Image.status == LabelStatus.VERIFIED
            ).count()
            prelabeled = Image.query.filter(
                Image.assigned_user_id == user_id,
                Image.status == LabelStatus.PRELABELED
            ).count()
            unlabeled = Image.query.filter(
                Image.assigned_user_id == user_id,
                Image.status == LabelStatus.UNLABELED
            ).count()

            print(f"  {config.USERS[user_id]:12} | Total: {total:5} | "
                  f"Verified: {verified:4} | Pre-labeled: {prelabeled:5} | "
                  f"Unlabeled: {unlabeled:5}")
            total_all += total

        # Check for any unassigned
        still_unassigned = Image.query.filter(Image.assigned_user_id.is_(None)).count()
        if still_unassigned > 0:
            print(f"\n  [WARNING] Still unassigned: {still_unassigned} images")

        print("-" * 50)
        print(f"  Total assigned: {total_all}")

        print("\n" + "=" * 60)
        print("Migration Complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Start the application: python -m labeling_tool.app")
        print("  2. Each user selects their name at http://localhost:5001")
        print("  3. Users label their assigned images")
        print("  4. Export JSON from dashboard")
        print("  5. Merge: python -m labeling_tool.scripts.merge_exports ...")
        print("=" * 60)


if __name__ == "__main__":
    migrate()
