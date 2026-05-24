"""Fast image-folder indexing for map-builder projects."""

from __future__ import annotations

from pathlib import Path

from .models import IndexSummary
from .project_store import PROJECT_DIR_NAME, ProjectStore


SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageIndexer:
    def __init__(self, read_dimensions: bool = True):
        self.read_dimensions = read_dimensions

    def index_folder(self, folder: Path, store: ProjectStore) -> IndexSummary:
        root = Path(folder).expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {root}")

        rel_paths: list[str] = []
        num_new = 0
        num_updated = 0
        num_unchanged = 0

        for path in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if path.name == PROJECT_DIR_NAME or not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                continue

            stat = path.stat()
            rel_path = path.relative_to(root).as_posix()
            width, height = self._read_dimensions(path) if self.read_dimensions else (None, None)
            result = store.upsert_image_index_entry(
                rel_path=rel_path,
                size_bytes=stat.st_size,
                mtime_ns=stat.st_mtime_ns,
                width=width,
                height=height,
            )
            rel_paths.append(rel_path)
            if result == "new":
                num_new += 1
            elif result == "updated":
                num_updated += 1
            else:
                num_unchanged += 1

        num_missing = store.mark_missing_except(rel_paths)
        return IndexSummary(
            num_found=len(rel_paths),
            num_new=num_new,
            num_updated=num_updated,
            num_missing=num_missing,
            num_unchanged=num_unchanged,
        )

    def _read_dimensions(self, path: Path) -> tuple[int | None, int | None]:
        try:
            import cv2  # type: ignore[import-not-found]
        except ImportError:
            return None, None

        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim < 2:
            return None, None
        height, width = image.shape[:2]
        return int(width), int(height)
