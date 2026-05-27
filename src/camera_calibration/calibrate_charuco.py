from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np


BOARD_SQUARES_X = 11
BOARD_SQUARES_Y = 8
SQUARE_LENGTH_M = 0.050
MARKER_LENGTH_M = 0.030

# Change this if the uploaded/printed board was generated with a different dictionary.
ARUCO_DICT_NAME = "DICT_4X4_50"

USE_LEGACY_PATTERN = False

MIN_CHARUCO_CORNERS = 8
MIN_VALID_IMAGES = 5
HEIF_EXTENSIONS = {".heic", ".heif"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", *HEIF_EXTENSIONS}

OUTPUT_YAML_NAME = "camera_calibration_charuco.yml"
OUTPUT_XML_NAME = "camera_params.xml"

_CV2: Any | None = None


@dataclass(frozen=True)
class DetectionResult:
    image_path: Path
    charuco_corners: np.ndarray
    charuco_ids: np.ndarray


def load_cv2_aruco() -> Any:
    global _CV2
    if _CV2 is not None:
        return _CV2

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "This utility requires OpenCV with the aruco module. "
            "Install opencv-contrib-python to use ChArUco calibration."
        ) from exc

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "This OpenCV build does not include cv2.aruco. "
            "Install opencv-contrib-python to use ChArUco calibration."
        )

    _CV2 = cv2
    return _CV2


def _get_aruco_dictionary() -> Any:
    cv2 = load_cv2_aruco()
    try:
        dict_id = getattr(cv2.aruco, ARUCO_DICT_NAME)
    except AttributeError as exc:
        raise ValueError(f"Unknown ArUco dictionary name: {ARUCO_DICT_NAME}") from exc

    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def _create_charuco_board(dictionary: Any) -> Any:
    cv2 = load_cv2_aruco()
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        board = cv2.aruco.CharucoBoard_create(
            BOARD_SQUARES_X,
            BOARD_SQUARES_Y,
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    else:
        board = cv2.aruco.CharucoBoard(
            (BOARD_SQUARES_X, BOARD_SQUARES_Y),
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )

    if hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(USE_LEGACY_PATTERN)
    return board


def _create_detector_parameters() -> Any:
    cv2 = load_cv2_aruco()
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        parameters = cv2.aruco.DetectorParameters_create()
    else:
        parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "CORNER_REFINE_SUBPIX") and hasattr(parameters, "cornerRefinementMethod"):
        parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    if hasattr(parameters, "cornerRefinementWinSize"):
        parameters.cornerRefinementWinSize = 5
    if hasattr(parameters, "cornerRefinementMaxIterations"):
        parameters.cornerRefinementMaxIterations = 30
    if hasattr(parameters, "cornerRefinementMinAccuracy"):
        parameters.cornerRefinementMinAccuracy = 0.01
    return parameters


def _create_charuco_detector(board: Any, parameters: Any) -> Any | None:
    cv2 = load_cv2_aruco()
    if not hasattr(cv2.aruco, "CharucoDetector"):
        return None

    try:
        detector = cv2.aruco.CharucoDetector(
            board,
            cv2.aruco.CharucoParameters(),
            parameters,
        )
    except (AttributeError, TypeError):
        detector = cv2.aruco.CharucoDetector(board)
        if hasattr(detector, "setDetectorParameters"):
            detector.setDetectorParameters(parameters)
    return detector


def _detect_markers_old_api(
    gray: np.ndarray,
    board: Any,
    dictionary: Any,
    parameters: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    cv2 = load_cv2_aruco()
    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
    if ids is None or len(corners) == 0:
        return None, None

    _refine_corners_subpixel(gray, corners)
    _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        corners,
        ids,
        gray,
        board,
    )
    return charuco_corners, charuco_ids


def _detect_markers_new_api(
    gray: np.ndarray,
    detector: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    return charuco_corners, charuco_ids


def _find_image_files(folder: Path) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _read_image(image_path: Path) -> np.ndarray | None:
    cv2 = load_cv2_aruco()
    if image_path.suffix.lower() not in HEIF_EXTENSIONS:
        return cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    try:
        from PIL import Image
        import pillow_heif
    except ImportError as exc:
        raise RuntimeError(
            "Reading HEIC/HEIF calibration images requires Pillow and pillow-heif. "
            "Install pillow pillow-heif, or convert the images to JPG or PNG."
        ) from exc

    pillow_heif.register_heif_opener()
    try:
        with Image.open(image_path) as pil_image:
            rgb_image = np.asarray(pil_image.convert("RGB"))
    except OSError:
        return None
    return cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)


def _refine_corners_subpixel(gray: np.ndarray, corners: list[np.ndarray]) -> None:
    cv2 = load_cv2_aruco()
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    for marker_corners in corners:
        cv2.cornerSubPix(gray, marker_corners, (5, 5), (-1, -1), criteria)


def _detect_charuco(
    image_path: Path,
    board: Any,
    dictionary: Any,
    parameters: Any,
    charuco_detector: Any | None,
) -> tuple[DetectionResult | None, tuple[int, int] | None]:
    cv2 = load_cv2_aruco()
    image = _read_image(image_path)
    if image is None:
        print(f"[WARN] Could not read image: {image_path}")
        return None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_size = (gray.shape[1], gray.shape[0])

    if charuco_detector is not None:
        charuco_corners, charuco_ids = _detect_markers_new_api(gray, charuco_detector)
    else:
        charuco_corners, charuco_ids = _detect_markers_old_api(
            gray,
            board,
            dictionary,
            parameters,
        )

    if charuco_ids is None or charuco_corners is None:
        return None, image_size
    if len(charuco_ids) < MIN_CHARUCO_CORNERS:
        return None, image_size

    _refine_corners_subpixel(gray, [charuco_corners])
    return DetectionResult(image_path, charuco_corners, charuco_ids), image_size


def _select_folder() -> Path | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askdirectory(title="Select folder containing ChArUco calibration images")
    root.destroy()
    if not selected:
        return None
    return Path(selected)


def _save_opencv_yaml(
    path: Path,
    image_size: tuple[int, int],
    rms_error: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    valid_image_count: int,
) -> None:
    cv2 = load_cv2_aruco()
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
    if not storage.isOpened():
        raise OSError(f"Could not open YAML file for writing: {path}")
    try:
        storage.write("model", "pinhole_radtan")
        storage.write("image_width", int(image_size[0]))
        storage.write("image_height", int(image_size[1]))
        storage.write("rms_reprojection_error", float(rms_error))
        storage.write("valid_image_count", int(valid_image_count))
        storage.write("board_squares_x", int(BOARD_SQUARES_X))
        storage.write("board_squares_y", int(BOARD_SQUARES_Y))
        storage.write("square_length_m", float(SQUARE_LENGTH_M))
        storage.write("marker_length_m", float(MARKER_LENGTH_M))
        storage.write("aruco_dictionary", ARUCO_DICT_NAME)
        storage.write("use_legacy_pattern", int(USE_LEGACY_PATTERN))
        storage.write("camera_matrix", camera_matrix)
        storage.write("distortion_coefficients", dist_coeffs.reshape(-1, 1))
    finally:
        storage.release()


def _save_map_builder_xml(
    path: Path,
    image_size: tuple[int, int],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> None:
    from map_builder.camera_models import PinholeRadTanCameraModel

    coeffs = np.zeros(5, dtype=float)
    flat_coeffs = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    coeffs[: min(5, flat_coeffs.size)] = flat_coeffs[:5]

    camera = PinholeRadTanCameraModel(
        image_width=int(image_size[0]),
        image_height=int(image_size[1]),
        fx=float(camera_matrix[0, 0]),
        fy=float(camera_matrix[1, 1]),
        cx=float(camera_matrix[0, 2]),
        cy=float(camera_matrix[1, 2]),
        k1=float(coeffs[0]),
        k2=float(coeffs[1]),
        p1=float(coeffs[2]),
        p2=float(coeffs[3]),
        k3=float(coeffs[4]),
    )
    camera.save_xml(path)


def _print_results(
    image_folder: Path,
    image_size: tuple[int, int],
    rms_error: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    detections: list[DetectionResult],
    yaml_path: Path,
    xml_path: Path,
) -> None:
    print()
    print("ChArUco camera calibration complete")
    print(f"Image folder: {image_folder}")
    print(f"Valid images: {len(detections)}")
    print(f"Image size: {image_size[0]} x {image_size[1]}")
    print(f"RMS reprojection error: {rms_error:.6f}")
    print()
    print("Camera matrix:")
    print(camera_matrix)
    print()
    print("Distortion coefficients [k1, k2, p1, p2, k3]:")
    print(np.asarray(dist_coeffs).reshape(-1)[:5])
    print()
    print(f"Saved OpenCV YAML: {yaml_path}")
    print(f"Saved camera XML: {xml_path}")


def calibrate_from_folder(image_folder: Path) -> tuple[float, np.ndarray, np.ndarray]:
    cv2 = load_cv2_aruco()
    dictionary = _get_aruco_dictionary()
    board = _create_charuco_board(dictionary)
    parameters = _create_detector_parameters()
    charuco_detector = _create_charuco_detector(board, parameters)
    image_paths = _find_image_files(image_folder)

    if not image_paths:
        raise ValueError(f"No calibration images found in {image_folder}")

    detections: list[DetectionResult] = []
    image_size: tuple[int, int] | None = None

    print(f"Scanning {len(image_paths)} image(s) in {image_folder}")
    for image_path in image_paths:
        detection, detected_size = _detect_charuco(
            image_path,
            board,
            dictionary,
            parameters,
            charuco_detector,
        )
        if detection is None:
            if detected_size is None:
                continue
            print(f"[SKIP] {image_path.name}: not enough ChArUco corners")
            continue

        if detected_size is None:
            print(f"[WARN] Skipping image with unknown size: {image_path}")
            continue
        if image_size is None:
            image_size = detected_size
        elif image_size != detected_size:
            print(f"[WARN] Skipping image with different size: {image_path}")
            continue

        detections.append(detection)
        print(f"[ OK ] {image_path.name}: {len(detection.charuco_ids)} ChArUco corners")

    if not detections:
        raise ValueError(
            f"No images contained at least {MIN_CHARUCO_CORNERS} ChArUco corners."
        )
    if len(detections) < MIN_VALID_IMAGES:
        raise ValueError(
            f"Found {len(detections)} valid calibration image(s), but at least "
            f"{MIN_VALID_IMAGES} are required. Take more images from varied viewpoints."
        )
    if image_size is None:
        raise ValueError("Internal error: accepted detections did not establish image size.")

    all_charuco_corners = [detection.charuco_corners for detection in detections]
    all_charuco_ids = [detection.charuco_ids for detection in detections]

    rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
        all_charuco_corners,
        all_charuco_ids,
        board,
        image_size,
        None,
        None,
    )

    yaml_path = image_folder / OUTPUT_YAML_NAME
    xml_path = image_folder / OUTPUT_XML_NAME
    _save_opencv_yaml(
        yaml_path,
        image_size,
        rms_error,
        camera_matrix,
        dist_coeffs,
        len(detections),
    )
    _save_map_builder_xml(xml_path, image_size, camera_matrix, dist_coeffs)
    _print_results(
        image_folder,
        image_size,
        rms_error,
        camera_matrix,
        dist_coeffs,
        detections,
        yaml_path,
        xml_path,
    )
    return rms_error, camera_matrix, dist_coeffs


def main() -> int:
    try:
        load_cv2_aruco()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    image_folder = _select_folder()
    if image_folder is None:
        print("No folder selected.")
        return 1

    try:
        calibrate_from_folder(image_folder)
    except Exception as exc:
        print(f"Calibration failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
