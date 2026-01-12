#!/usr/bin/env python3
"""
Build unified image index across all datasets.

Usage:
    python -m labeling_tool.scripts.index_datasets [--force]
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description='Index all datasets for labeling'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force reindex (clear existing records)'
    )
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        help='Index only specific dataset'
    )
    args = parser.parse_args()

    # Create Flask app context
    from labeling_tool.app import create_app
    app = create_app()

    with app.app_context():
        from labeling_tool.services.indexer import ImageIndexer
        from labeling_tool import config

        print("\n" + "=" * 50)
        print("ResQPet Image Indexer")
        print("=" * 50)

        if args.dataset:
            # Index single dataset
            if args.dataset not in config.DATASETS:
                print(f"Unknown dataset: {args.dataset}")
                print(f"Available: {list(config.DATASETS.keys())}")
                return 1

            indexer = ImageIndexer()
            count = indexer.index_dataset(
                args.dataset,
                str(config.DATASETS[args.dataset]),
                force_reindex=args.force
            )
            print(f"\nIndexed {count} images from {args.dataset}")

        else:
            # Index all datasets
            indexer = ImageIndexer()
            results = indexer.index_all(force_reindex=args.force)

            print("\n" + "=" * 50)
            print("Indexing Complete")
            print("=" * 50)

            for dataset, count in results.items():
                if dataset != 'total':
                    print(f"  {dataset}: {count} images")

            print("-" * 50)
            print(f"  TOTAL: {results['total']} images")
            print("=" * 50)

    return 0


if __name__ == '__main__':
    sys.exit(main())
