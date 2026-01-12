#!/usr/bin/env python3
"""
Migration script: Convert absolute paths to relative paths in database.

This script converts:
- Dataset.base_path: from absolute to relative (relative to PROJECT_ROOT)
- Image.original_path: from absolute to relative (relative to PROJECT_ROOT)

After migration, paths will be reconstructed at runtime using:
- Dataset.absolute_base_path property
- Image.absolute_path property
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from labeling_tool.app import create_app
from labeling_tool import config
from labeling_tool.database import db
from labeling_tool.database.models import Dataset, Image


def migrate_dataset_paths():
    """Convert Dataset.base_path from absolute to relative."""
    datasets = Dataset.query.all()
    converted = 0
    skipped = 0

    print(f"\n[Migration] Processing {len(datasets)} datasets...")

    for dataset in datasets:
        old_path = dataset.base_path
        path_obj = Path(old_path)

        # Check if already relative
        if not path_obj.is_absolute():
            print(f"  [SKIP] {dataset.name}: already relative ({old_path})")
            skipped += 1
            continue

        # Try to make relative to PROJECT_ROOT
        try:
            relative_path = path_obj.relative_to(config.PROJECT_ROOT)
            dataset.base_path = str(relative_path)
            print(f"  [OK] {dataset.name}:")
            print(f"        OLD: {old_path}")
            print(f"        NEW: {relative_path}")
            converted += 1
        except ValueError:
            # Path is not under PROJECT_ROOT - keep as is
            print(f"  [WARN] {dataset.name}: not under PROJECT_ROOT, keeping absolute")
            print(f"         Path: {old_path}")
            skipped += 1

    return converted, skipped


def migrate_image_paths():
    """Convert Image.original_path from absolute to relative."""
    total = Image.query.count()
    print(f"\n[Migration] Processing {total} images...")

    converted = 0
    skipped = 0
    errors = 0
    batch_size = 500

    # Process in batches
    offset = 0
    while offset < total:
        images = Image.query.offset(offset).limit(batch_size).all()

        for image in images:
            if not image.original_path:
                skipped += 1
                continue

            old_path = image.original_path
            path_obj = Path(old_path)

            # Check if already relative
            if not path_obj.is_absolute():
                skipped += 1
                continue

            # Try to make relative to PROJECT_ROOT
            try:
                relative_path = path_obj.relative_to(config.PROJECT_ROOT)
                image.original_path = str(relative_path)
                converted += 1
            except ValueError:
                # Path is not under PROJECT_ROOT
                # Try to extract relative path from dataset base
                if image.dataset:
                    try:
                        dataset_base = Path(image.dataset.base_path)
                        if dataset_base.is_absolute():
                            relative_path = path_obj.relative_to(dataset_base)
                            # Store as dataset_relative_base/relative_path
                            try:
                                dataset_rel = dataset_base.relative_to(config.PROJECT_ROOT)
                                image.original_path = str(dataset_rel / relative_path)
                                converted += 1
                                continue
                            except ValueError:
                                pass
                    except ValueError:
                        pass

                errors += 1
                if errors <= 10:
                    print(f"  [ERROR] Image {image.id}: Cannot convert {old_path}")

        # Commit batch
        db.session.commit()
        offset += batch_size

        # Progress
        progress = min(offset, total)
        print(f"  Processed {progress}/{total} ({progress*100//total}%)")

    return converted, skipped, errors


def verify_migration():
    """Verify that paths can be reconstructed correctly."""
    print("\n[Verification] Checking path reconstruction...")

    # Check datasets
    datasets = Dataset.query.all()
    dataset_ok = 0
    dataset_fail = 0

    for dataset in datasets:
        abs_path = dataset.absolute_base_path
        if abs_path.exists():
            dataset_ok += 1
        else:
            dataset_fail += 1
            print(f"  [FAIL] Dataset '{dataset.name}': {abs_path} does not exist")

    print(f"  Datasets: {dataset_ok} OK, {dataset_fail} FAIL")

    # Check sample of images
    sample_size = min(100, Image.query.count())
    images = Image.query.limit(sample_size).all()
    image_ok = 0
    image_fail = 0

    for image in images:
        try:
            abs_path = image.absolute_path
            if abs_path.exists():
                image_ok += 1
            else:
                image_fail += 1
                if image_fail <= 5:
                    print(f"  [FAIL] Image {image.id}: {abs_path} does not exist")
        except Exception as e:
            image_fail += 1
            if image_fail <= 5:
                print(f"  [FAIL] Image {image.id}: {e}")

    print(f"  Images (sample of {sample_size}): {image_ok} OK, {image_fail} FAIL")

    return dataset_fail == 0 and image_fail == 0


def main():
    """Run migration."""
    print("=" * 60)
    print("PATH MIGRATION: Absolute -> Relative")
    print("=" * 60)
    print(f"\nPROJECT_ROOT: {config.PROJECT_ROOT}")

    app = create_app()

    with app.app_context():
        # Show current state
        print("\n[Current State]")
        dataset_sample = Dataset.query.first()
        if dataset_sample:
            print(f"  Dataset example: {dataset_sample.base_path}")
        image_sample = Image.query.first()
        if image_sample:
            print(f"  Image example: {image_sample.original_path}")

        # Confirm migration
        print("\n" + "-" * 60)
        response = input("Proceed with migration? (yes/no): ")
        if response.lower() != 'yes':
            print("Migration cancelled.")
            return

        # Migrate datasets
        ds_converted, ds_skipped = migrate_dataset_paths()
        print(f"\n[Dataset Results] Converted: {ds_converted}, Skipped: {ds_skipped}")

        # Commit dataset changes
        db.session.commit()

        # Migrate images
        img_converted, img_skipped, img_errors = migrate_image_paths()
        print(f"\n[Image Results] Converted: {img_converted}, Skipped: {img_skipped}, Errors: {img_errors}")

        # Final commit
        db.session.commit()

        # Verify
        print("\n" + "-" * 60)
        success = verify_migration()

        # Summary
        print("\n" + "=" * 60)
        print("MIGRATION COMPLETE")
        print("=" * 60)
        print(f"Datasets converted: {ds_converted}")
        print(f"Images converted: {img_converted}")
        print(f"Verification: {'PASSED' if success else 'FAILED'}")

        if not success:
            print("\n[WARNING] Some paths could not be verified.")
            print("Check the errors above and ensure datasets exist at the expected locations.")


if __name__ == '__main__':
    main()
