"""
Image Indexer Service
Indexes all images from configured datasets into the database
"""

from pathlib import Path
from PIL import Image as PILImage
from tqdm import tqdm

from labeling_tool.database import db
from labeling_tool.database.models import Dataset, Image
from labeling_tool import config


class ImageIndexer:
    """Index all images from configured datasets."""

    def __init__(self, datasets_config: dict = None):
        """
        Initialize indexer with dataset configuration.

        Args:
            datasets_config: Dict mapping dataset names to paths.
                            If None, uses config.DATASETS
        """
        self.datasets_config = datasets_config or config.DATASETS
        self.supported_extensions = config.SUPPORTED_EXTENSIONS

    def index_all(self, force_reindex=False):
        """
        Index all configured datasets.

        Args:
            force_reindex: If True, clear existing records and reindex

        Returns:
            Dict with counts per dataset
        """
        results = {}
        total = 0

        for name, base_path in self.datasets_config.items():
            count = self.index_dataset(name, str(base_path), force_reindex)
            results[name] = count
            total += count

        results['total'] = total
        return results

    def index_dataset(self, name: str, base_path: str, force_reindex=False):
        """
        Index a single dataset.

        Args:
            name: Dataset name
            base_path: Path to dataset directory
            force_reindex: If True, clear existing records

        Returns:
            Number of images indexed
        """
        base_path = Path(base_path)

        if not base_path.exists():
            print(f"[Indexer] Dataset path not found: {base_path}")
            return 0

        # Calcola path relativo a PROJECT_ROOT per portabilità
        try:
            relative_base_path = str(base_path.relative_to(config.PROJECT_ROOT))
        except ValueError:
            # Se non è sotto PROJECT_ROOT, usa path assoluto (fallback)
            relative_base_path = str(base_path)

        # Get or create dataset record
        dataset = Dataset.query.filter_by(name=name).first()

        if dataset is None:
            dataset = Dataset(name=name, base_path=relative_base_path)  # PATH RELATIVO
            db.session.add(dataset)
            db.session.commit()
            print(f"[Indexer] Created dataset: {name} (base_path: {relative_base_path})")

        elif force_reindex:
            # Clear existing images for reindex
            deleted = Image.query.filter_by(dataset_id=dataset.id).delete()
            db.session.commit()
            print(f"[Indexer] Deleted {deleted} existing images for reindex")

        # Find all images
        image_paths = []
        for ext in self.supported_extensions:
            image_paths.extend(base_path.rglob(f'*{ext}'))
            image_paths.extend(base_path.rglob(f'*{ext.upper()}'))

        # Deduplicate
        image_paths = list(set(image_paths))

        print(f"[Indexer] Found {len(image_paths)} images in {name}")

        # Index each image
        indexed = 0
        errors = 0

        for img_path in tqdm(image_paths, desc=f"Indexing {name}"):
            try:
                if self._index_image(dataset, img_path, base_path):
                    indexed += 1
            except Exception as e:
                errors += 1
                if errors <= 5:
                    print(f"[Indexer] Error indexing {img_path}: {e}")

            # Commit in batches
            if indexed % 500 == 0:
                db.session.commit()

        # Final commit
        db.session.commit()

        # Update dataset count
        dataset.image_count = Image.query.filter_by(dataset_id=dataset.id).count()
        db.session.commit()

        print(f"[Indexer] Indexed {indexed} images, {errors} errors")
        return indexed

    def _index_image(self, dataset: Dataset, img_path: Path, base_path: Path) -> bool:
        """
        Index a single image.

        Returns:
            True if newly indexed, False if already exists
        """
        # Calculate relative path from base_path
        relative_path = str(img_path.relative_to(base_path))

        # Check if already indexed (by dataset + relative_path)
        existing = Image.query.filter_by(
            dataset_id=dataset.id,
            relative_path=relative_path
        ).first()

        if existing:
            return False

        # Get image metadata
        try:
            with PILImage.open(img_path) as img:
                width, height = img.size
        except Exception:
            # Can't read image dimensions, use defaults
            width, height = None, None

        file_size = img_path.stat().st_size

        # Salva path relativo a PROJECT_ROOT (per retrocompatibilità con original_path)
        try:
            original_path_relative = str(img_path.relative_to(config.PROJECT_ROOT))
        except ValueError:
            original_path_relative = str(img_path)  # Fallback assoluto

        # Create image record
        image = Image(
            dataset_id=dataset.id,
            original_path=original_path_relative,  # Ora relativo!
            relative_path=relative_path,
            filename=img_path.name,
            width=width,
            height=height,
            file_size=file_size
        )

        db.session.add(image)
        return True


def index_all_datasets(force_reindex=False):
    """
    Convenience function to index all datasets.

    Args:
        force_reindex: If True, clear existing records

    Returns:
        Dict with counts per dataset
    """
    indexer = ImageIndexer()
    return indexer.index_all(force_reindex)
