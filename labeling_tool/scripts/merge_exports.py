#!/usr/bin/env python3
"""
Merge user JSON exports into final YOLO format.

Usage:
    python -m labeling_tool.scripts.merge_exports \
        --json-dir labeling_data/exports/json \
        --output labeling_data/exports/merged_yolo \
        --train-split 0.8
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


def load_user_exports(json_dir: Path) -> dict:
    """
    Load all user JSON exports from directory.

    Returns:
        dict: {user_id: export_data}
    """
    exports = {}

    json_files = list(json_dir.glob("user_*_annotations_*.json"))
    print(f"[INFO] Found {len(json_files)} JSON export files")

    for json_file in json_files:
        print(f"  - Loading: {json_file.name}")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        user_id = data["export_metadata"]["user_id"]

        # Keep latest version per user
        if user_id not in exports:
            exports[user_id] = data
        else:
            existing_version = int(exports[user_id]["export_metadata"]["version"].replace("v", ""))
            new_version = int(data["export_metadata"]["version"].replace("v", ""))
            if new_version > existing_version:
                print(f"    -> Updating to newer version v{new_version}")
                exports[user_id] = data

    return exports


def merge_annotations(exports: dict) -> tuple:
    """
    Merge annotations from all users.

    Handles:
    - Deduplication by image_id
    - Conflict resolution (if same image labeled differently)

    Returns:
        tuple: (merged_list, conflicts_list)
    """
    merged = {}
    conflicts = []

    for user_id, data in exports.items():
        user_name = data["export_metadata"]["user_name"]
        print(f"[INFO] Processing {user_name}: {len(data['annotations'])} annotations")

        for ann in data["annotations"]:
            image_id = ann["image_id"]

            if image_id in merged:
                # Conflict: same image in multiple exports
                existing = merged[image_id]
                if existing["label_value"] != ann["label_value"]:
                    conflicts.append({
                        "image_id": image_id,
                        "existing_user": existing.get("_user_id"),
                        "existing_label": existing["label"],
                        "new_user": user_id,
                        "new_label": ann["label"]
                    })
                # Keep existing (first wins)
            else:
                ann["_user_id"] = user_id
                ann["_user_name"] = user_name
                merged[image_id] = ann

    return list(merged.values()), conflicts


def export_yolo(annotations: list, output_dir: Path, train_split: float = 0.8) -> dict:
    """Export merged annotations to YOLO format."""

    # Clean output directory if exists
    if output_dir.exists():
        print(f"[WARN] Output directory exists, removing: {output_dir}")
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create directories
    (output_dir / 'images' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'images' / 'val').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'train').mkdir(parents=True, exist_ok=True)
    (output_dir / 'labels' / 'val').mkdir(parents=True, exist_ok=True)

    # Filter valid labels (0 = WITH_COLLAR, 1 = WITHOUT_COLLAR only)
    valid_annotations = [a for a in annotations if a["label_value"] in [0, 1]]
    skipped = len(annotations) - len(valid_annotations)
    if skipped > 0:
        print(f"[INFO] Skipping {skipped} annotations with UNCLEAR/NO_DOG labels")

    # Shuffle and split
    random.shuffle(valid_annotations)
    split_idx = int(len(valid_annotations) * train_split)
    train_set = valid_annotations[:split_idx]
    val_set = valid_annotations[split_idx:]

    stats = {"train": 0, "val": 0, "errors": 0, "users": set()}

    for split_name, split_data in [("train", train_set), ("val", val_set)]:
        print(f"[INFO] Processing {split_name} set: {len(split_data)} images")

        for ann in split_data:
            try:
                image_path = Path(ann["image_path"])
                if not image_path.exists():
                    print(f"  [SKIP] Image not found: {image_path}")
                    stats["errors"] += 1
                    continue

                # Create symlink with unique name
                link_name = image_path.name
                img_link = output_dir / 'images' / split_name / link_name

                # Handle duplicates
                counter = 1
                while img_link.exists():
                    stem = image_path.stem
                    suffix = image_path.suffix
                    img_link = output_dir / 'images' / split_name / f"{stem}_{counter}{suffix}"
                    link_name = img_link.name
                    counter += 1

                # Use absolute path for symlink to avoid broken links
                os.symlink(image_path.resolve(), img_link)

                # Create label file
                label_file = output_dir / 'labels' / split_name / (img_link.stem + '.txt')
                class_id = ann["label_value"]

                # Parse bbox
                if ann.get("dog_bbox"):
                    bbox = ann["dog_bbox"]
                    x1, y1, x2, y2 = bbox
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                else:
                    # Full image if no bbox
                    cx, cy, w, h = 0.5, 0.5, 1.0, 1.0

                with open(label_file, 'w') as f:
                    f.write(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

                stats[split_name] += 1
                if ann.get("_user_id"):
                    stats["users"].add(ann["_user_id"])

            except Exception as e:
                print(f"  [ERROR] {ann['image_path']}: {e}")
                stats["errors"] += 1

    # Create data.yaml
    data_yaml_content = f"""# ResQPet Collar Detection Dataset
# Merged from {len(stats['users'])} users on {datetime.utcnow().isoformat()}

path: {output_dir.absolute()}
train: images/train
val: images/val

nc: 2
names:
  0: Dog-with-Leash
  1: Dog-without-Leash
"""

    with open(output_dir / 'data.yaml', 'w') as f:
        f.write(data_yaml_content)

    # Convert users set to list for JSON serialization
    stats["users"] = list(stats["users"])

    return stats


def create_merge_report(output_dir: Path, stats: dict, conflicts: list, exports: dict):
    """Create a detailed merge report."""
    report = {
        "merge_date": datetime.utcnow().isoformat() + "Z",
        "train_images": stats["train"],
        "val_images": stats["val"],
        "total_images": stats["train"] + stats["val"],
        "errors": stats["errors"],
        "source_users": stats["users"],
        "conflicts_count": len(conflicts),
        "conflicts": conflicts[:20] if conflicts else [],  # First 20 conflicts
        "user_contributions": {}
    }

    # Calculate per-user contributions
    for user_id, data in exports.items():
        user_name = data["export_metadata"]["user_name"]
        report["user_contributions"][user_name] = {
            "user_id": user_id,
            "total_annotations": len(data["annotations"]),
            "export_date": data["export_metadata"]["export_date"],
            "version": data["export_metadata"]["version"]
        }

    with open(output_dir / 'merge_report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Merge user JSON exports to YOLO format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m labeling_tool.scripts.merge_exports --json-dir exports/json --output exports/yolo_merged

  python -m labeling_tool.scripts.merge_exports \\
      --json-dir labeling_data/exports/json \\
      --output labeling_data/exports/collar_labels_merged \\
      --train-split 0.85
        """
    )
    parser.add_argument(
        "--json-dir",
        required=True,
        help="Directory containing user JSON exports"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for YOLO dataset"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="Train/val split ratio (default: 0.8)"
    )

    args = parser.parse_args()

    json_dir = Path(args.json_dir)
    output_dir = Path(args.output)

    print("=" * 60)
    print("ResQPet JSON to YOLO Merge Tool")
    print("=" * 60)

    # Validate input
    if not json_dir.exists():
        print(f"[ERROR] JSON directory not found: {json_dir}")
        return 1

    # Load exports
    print(f"\n[Step 1] Loading exports from: {json_dir}")
    exports = load_user_exports(json_dir)

    if not exports:
        print("[ERROR] No valid exports found!")
        return 1

    print(f"\n[INFO] Found exports from {len(exports)} users:")
    for user_id, data in exports.items():
        print(f"  - {data['export_metadata']['user_name']}: "
              f"{len(data['annotations'])} annotations "
              f"({data['export_metadata']['version']})")

    # Merge
    print(f"\n[Step 2] Merging annotations...")
    merged, conflicts = merge_annotations(exports)
    print(f"[INFO] Total merged annotations: {len(merged)}")

    if conflicts:
        print(f"\n[WARNING] Found {len(conflicts)} conflicts:")
        for c in conflicts[:5]:  # Show first 5
            print(f"  Image {c['image_id']}: User {c['existing_user']} ({c['existing_label']}) "
                  f"vs User {c['new_user']} ({c['new_label']})")
        if len(conflicts) > 5:
            print(f"  ... and {len(conflicts) - 5} more")

    # Export
    print(f"\n[Step 3] Exporting to YOLO format: {output_dir}")
    stats = export_yolo(merged, output_dir, args.train_split)

    # Create report
    print(f"\n[Step 4] Creating merge report...")
    report = create_merge_report(output_dir, stats, conflicts, exports)

    # Summary
    print("\n" + "=" * 60)
    print("MERGE COMPLETE!")
    print("=" * 60)
    print(f"  Train images: {stats['train']}")
    print(f"  Val images:   {stats['val']}")
    print(f"  Total:        {stats['train'] + stats['val']}")
    print(f"  Errors:       {stats['errors']}")
    print(f"  Conflicts:    {len(conflicts)}")
    print(f"\n  Output:       {output_dir}")
    print(f"  data.yaml:    {output_dir / 'data.yaml'}")
    print(f"  Report:       {output_dir / 'merge_report.json'}")
    print("=" * 60)

    print("\nTo train with this dataset:")
    print(f"  yolo detect train data={output_dir / 'data.yaml'} model=yolov8n.pt epochs=100")

    return 0


if __name__ == "__main__":
    exit(main())
