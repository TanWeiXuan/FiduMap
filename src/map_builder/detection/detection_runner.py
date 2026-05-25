"""Run marker detection over indexed project images."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from map_builder.project.models import DetectorRunConfig, ImageRecord
from map_builder.project.project_store import ProjectStore

from .marker_detector import marker_family_for_dictionary
from .opencv_aruco_detector import OpenCVArucoMarkerDetector


try:
    import cv2 as _CV2  # type: ignore[import-not-found]
except ImportError:
    _CV2 = None


ProgressCallback = Callable[[int, int, ImageRecord, str], None]


class DetectionRunner:
    def __init__(
        self,
        project_folder: Path,
        store: ProjectStore,
        detector_config: DetectorRunConfig,
        progress_callback: ProgressCallback | None = None,
    ):
        self.project_folder = Path(project_folder)
        self.store = store
        self.detector_config = detector_config
        self.progress_callback = progress_callback

    def run(self) -> int:
        _load_cv2()
        family = self.detector_config.marker_family or marker_family_for_dictionary(
            self.detector_config.dictionary_name,
            self.detector_config.detector_type,
        )
        detector = OpenCVArucoMarkerDetector(
            dictionary_name=self.detector_config.dictionary_name,
            marker_family=family,
            corner_refinement=str(self.detector_config.detector_params.get("corner_refinement", "auto")),
        )
        run_id = self.store.create_detector_run(self.detector_config, opencv_version=getattr(_CV2, "__version__", None))

        images = self.store.list_detectable_images()
        total = len(images)
        for index, image in enumerate(images, start=1):
            self._notify(index, total, image, "loading")
            path = image.absolute_path(self.project_folder)
            data = _CV2.imread(str(path), _CV2.IMREAD_COLOR)
            if data is None:
                self._notify(index, total, image, "failed_to_read")
                continue
            self._notify(index, total, image, "detecting")
            detections = detector.detect(data)
            self.store.replace_image_detections(image.id, run_id, detections)
            self._notify(index, total, image, f"{len(detections)} markers")
        return total

    def _notify(self, index: int, total: int, image: ImageRecord, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(index, total, image, message)


def _load_cv2() -> Any:
    if _CV2 is None:
        raise RuntimeError("OpenCV is required for marker detection. Install opencv-contrib-python.")
    return _CV2
