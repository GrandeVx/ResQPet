#!/usr/bin/env python3
"""
Export verified labels to YOLO format for retraining.

Usage:
    python -m labeling_tool.scripts.export_yolo [--train-split 0.8]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Export verified labels to YOLO format'
    )
    parser.add_argument(
        '--train-split', '-t',
        type=float,
        default=0.8,
        help='Fraction for training set (default: 0.8)'
    )
    parser.add_argument(
        '--min-confidence', '-c',
        type=float,
        default=0.0,
        help='Minimum confidence for model-only labels (default: 0.0)'
    )
    parser.add_argument(
        '--include-model-only', '-m',
        action='store_true',
        help='Include pre-labeled images (not just human-verified)'
    )
    parser.add_argument(
        '--version', '-v',
        type=int,
        default=None,
        help='Export version number (default: auto-increment)'
    )
    args = parser.parse_args()

    # Create Flask app context
    from labeling_tool.app import create_app
    app = create_app()

    with app.app_context():
        from labeling_tool.services.exporter import YOLOExporter
        from labeling_tool import config

        print("\n" + "=" * 50)
        print("ResQPet YOLO Export")
        print("=" * 50)

        exporter = YOLOExporter(str(config.EXPORTS_DIR))

        result = exporter.export(
            train_split=args.train_split,
            min_confidence=args.min_confidence,
            include_model_only=args.include_model_only,
            version=args.version
        )

        print("\n" + "=" * 50)
        print("Export Complete")
        print("=" * 50)
        print(f"  Version: v{result['version']}")
        print(f"  Path: {result['path']}")
        print(f"  Train images: {result['train_images']}")
        print(f"  Val images: {result['val_images']}")
        print(f"  Total: {result['total']}")
        print("=" * 50)
        print("\nTo train with this dataset:")
        print(f"  yolo detect train data={result['path']}/data.yaml")
        print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())
