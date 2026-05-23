"""XML I/O for camera models using xml.etree.ElementTree only."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from .omni_radtan import OmniRadTanCameraModel
from .pinhole_radtan import PinholeRadTanCameraModel


def _require_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        raise ValueError(f"Missing required XML field: <{tag}>.")
    return child


def _require_text(parent: ET.Element, tag: str) -> str:
    node = _require_child(parent, tag)
    text = node.text
    if text is None or text.strip() == "":
        raise ValueError(f"Missing required text for XML field: <{tag}>.")
    return text.strip()


def _require_int(parent: ET.Element, tag: str) -> int:
    text = _require_text(parent, tag)
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for XML field <{tag}>.") from exc


def _require_float(parent: ET.Element, tag: str) -> float:
    text = _require_text(parent, tag)
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"Invalid float for XML field <{tag}>.") from exc


def save_camera_model_xml(camera: PinholeRadTanCameraModel | OmniRadTanCameraModel, path: str | Path) -> None:
    root = ET.Element("camera")
    ET.SubElement(root, "model").text = camera.model_name
    ET.SubElement(root, "image_width").text = str(camera.image_width)
    ET.SubElement(root, "image_height").text = str(camera.image_height)

    intr = ET.SubElement(root, "intrinsics")
    ET.SubElement(intr, "fx").text = str(camera.fx)
    ET.SubElement(intr, "fy").text = str(camera.fy)
    ET.SubElement(intr, "cx").text = str(camera.cx)
    ET.SubElement(intr, "cy").text = str(camera.cy)

    if isinstance(camera, OmniRadTanCameraModel):
        omni = ET.SubElement(root, "omni")
        ET.SubElement(omni, "xi").text = str(camera.xi)

    dist = ET.SubElement(root, "distortion")
    ET.SubElement(dist, "k1").text = str(camera.k1)
    ET.SubElement(dist, "k2").text = str(camera.k2)
    ET.SubElement(dist, "p1").text = str(camera.p1)
    ET.SubElement(dist, "p2").text = str(camera.p2)
    ET.SubElement(dist, "k3").text = str(camera.k3)

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)


def load_camera_model_xml(path: str | Path) -> PinholeRadTanCameraModel | OmniRadTanCameraModel:
    root = ET.parse(Path(path)).getroot()
    model = _require_text(root, "model")

    width = _require_int(root, "image_width")
    height = _require_int(root, "image_height")
    intr = _require_child(root, "intrinsics")
    dist = _require_child(root, "distortion")

    common = dict(
        image_width=width,
        image_height=height,
        fx=_require_float(intr, "fx"),
        fy=_require_float(intr, "fy"),
        cx=_require_float(intr, "cx"),
        cy=_require_float(intr, "cy"),
        k1=_require_float(dist, "k1"),
        k2=_require_float(dist, "k2"),
        p1=_require_float(dist, "p1"),
        p2=_require_float(dist, "p2"),
        k3=_require_float(dist, "k3"),
    )

    if model == "pinhole_radtan":
        return PinholeRadTanCameraModel(**common)
    if model == "omni_radtan":
        omni = _require_child(root, "omni")
        xi = _require_float(omni, "xi")
        return OmniRadTanCameraModel(xi=xi, **common)
    raise ValueError(f"Unknown camera model '{model}'.")


def write_pinhole_radtan_example_xml(path: str | Path) -> None:
    PinholeRadTanCameraModel.write_example_xml(path)


def write_omni_radtan_example_xml(path: str | Path) -> None:
    OmniRadTanCameraModel.write_example_xml(path)
