#!/usr/bin/env python3
"""
Run batch pre-labeling on indexed images using trained models.

Usage:
    python -m labeling_tool.scripts.run_prelabeling [--batch-size 100]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Run pre-labeling using trained models'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=100,
        help='Commit after this many images (default: 100)'
    )
    parser.add_argument(
        '--include-existing',
        action='store_true',
        help='Re-process images that already have annotations'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Process only this many images (for testing)'
    )
    args = parser.parse_args()

    # Create Flask app context
    from labeling_tool.app import create_app
    app = create_app()

    with app.app_context():
        from labeling_tool.services.prelabeler import PreLabeler
        from labeling_tool.database.models import Image, LabelStatus

        print("\n" + "=" * 50)
        print("ResQPet Pre-labeling")
        print("=" * 50)

        # Check how many images need processing
        query = Image.query.filter(Image.status == LabelStatus.UNLABELED)
        if not args.include_existing:
            query = query.filter(Image.collar_label == None)

        pending = query.count()
        print(f"\nImages pending pre-labeling: {pending}")

        if pending == 0:
            print("No images to process!")
            return 0

        if args.limit:
            print(f"Limiting to {args.limit} images")

        # Initialize pre-labeler
        prelabeler = PreLabeler()

        # Process images
        results = prelabeler.prelabel_all(
            batch_size=args.batch_size,
            skip_existing=not args.include_existing
        )

        print("\n" + "=" * 50)
        print("Pre-labeling Complete")
        print("=" * 50)
        print(f"  Processed: {results['processed']}")
        print(f"  No dog detected: {results.get('no_dog', 0)}")
        print(f"  Errors: {results['errors']}")
        print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())
