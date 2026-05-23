from pathlib import Path

import pytest

from map_builder.camera_models import (
    OmniRadTanCameraModel,
    PinholeRadTanCameraModel,
    load_camera_model_xml,
    write_omni_radtan_example_xml,
    write_pinhole_radtan_example_xml,
)


def test_write_and_load_pinhole_example_xml(tmp_path: Path) -> None:
    path = tmp_path / "pin.xml"
    write_pinhole_radtan_example_xml(path)
    cam = load_camera_model_xml(path)
    assert isinstance(cam, PinholeRadTanCameraModel)
    assert cam.image_width == 1920
    assert cam.image_height == 1080
    assert cam.fx == 1000.0


def test_write_and_load_omni_example_xml(tmp_path: Path) -> None:
    path = tmp_path / "omni.xml"
    write_omni_radtan_example_xml(path)
    cam = load_camera_model_xml(path)
    assert isinstance(cam, OmniRadTanCameraModel)
    assert cam.image_width == 1920
    assert cam.image_height == 1080
    assert cam.xi == 1.0


def test_missing_required_field_raises(tmp_path: Path) -> None:
    path = tmp_path / "bad.xml"
    path.write_text("<camera><model>pinhole_radtan</model></camera>", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing required XML field"):
        load_camera_model_xml(path)


def test_unknown_model_raises(tmp_path: Path) -> None:
    path = tmp_path / "unknown.xml"
    path.write_text(
        """
<camera>
  <model>unknown</model>
  <image_width>10</image_width>
  <image_height>10</image_height>
  <intrinsics><fx>1</fx><fy>1</fy><cx>0</cx><cy>0</cy></intrinsics>
  <distortion><k1>0</k1><k2>0</k2><p1>0</p1><p2>0</p2><k3>0</k3></distortion>
</camera>
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Unknown camera model"):
        load_camera_model_xml(path)
