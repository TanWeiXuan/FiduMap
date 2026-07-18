import importlib

import cv2
import numpy as np
import pytest

from map_builder.metric_depth.export import export_artifact
from map_builder.metric_depth.models import MetricDepthArtifact, MetricDepthMetrics
from map_builder.metric_depth.store import MetricDepthStore


def _artifact(image_id=2, distance=2.0):
    z = np.full((3, 4), distance, dtype=np.float32)
    valid = np.ones((3, 4), dtype=bool)
    return MetricDepthArtifact(image_id, "fake", 4, 3, z, z.copy(), valid, np.full_like(z, 0.75), np.where(valid, 1.5, 0).astype(np.float32), valid.copy(), {"test": True}, MetricDepthMetrics(prompt_pixel_count=12, prompt_spatial_coverage=0.5, valid_output_fraction=1.0, status="success"))


def test_atomic_npz_and_metadata_round_trip_and_staleness(tmp_path):
    with MetricDepthStore.open(tmp_path) as store:
        run = store.create_run("fake", "local", {}, 7, "points=0")
        path = store.save_artifact_atomic(run, _artifact())
        assert path.exists()
        loaded = store.load_artifact(run, 2)
        assert loaded is not None and np.allclose(loaded.range_m, 2.0)
        assert store.latest_successful_record(2, 7) is not None
        assert store.latest_successful_record(2, 8) is None
        assert store.counts(8)["stale"] == 1


def test_portable_uint16_export_and_overflow_warning(tmp_path):
    artifact = _artifact(distance=2.345)
    paths = export_artifact(artifact, tmp_path)
    range_mm = cv2.imread(str(paths["range_mm"]), cv2.IMREAD_UNCHANGED)
    assert range_mm.dtype == np.uint16 and range_mm[0, 0] == 2345
    overflow = _artifact(distance=70.0)
    overflow_paths = export_artifact(overflow, tmp_path, stem="overflow")
    metadata = overflow_paths["metadata"].read_text(encoding="utf-8")
    assert '"portable_range_mm_overflow": true' in metadata
    assert cv2.imread(str(overflow_paths["range_mm"]), cv2.IMREAD_UNCHANGED)[0, 0] == 0


def test_gui_imports_without_importing_optional_libraries(monkeypatch):
    import sys
    before = set(sys.modules)
    importlib.import_module("map_builder.gui.metric_depth_control_panel")
    importlib.import_module("map_builder.gui.depth_3d_viewer_panel")
    importlib.import_module("map_builder.gui.right_panel_tabs")
    newly_loaded = set(sys.modules) - before
    assert "transformers" not in newly_loaded
    assert not any(name == "vtkmodules" or name.startswith("vtkmodules.") for name in newly_loaded)


def test_depth_display_modes_are_deterministic():
    from map_builder.gui.image_viewer_panel import render_depth_display
    source = np.zeros((3, 4, 3), dtype=np.uint8)
    artifact = _artifact()
    first, limits = render_depth_display(source, artifact, "Metric range")
    second, _ = render_depth_display(source, artifact, "Metric range")
    assert np.array_equal(first, second)
    assert "m range" in limits


def test_gui_tab_order_numeric_validation_progress_and_placeholder():
    import tkinter as tk
    from map_builder.gui.main_window import MainWindow
    from map_builder.metric_depth.models import MetricDepthProgress

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tk display unavailable")
    try:
        root.withdraw()
        window = MainWindow(root)
        assert [window.workflow_tabs.tab(i, "text") for i in range(window.workflow_tabs.index("end"))] == ["Marker BA Pipeline", "Dense Reconstruction", "Metric Depth Maps"]
        assert [window.right_tabs.tab(i, "text") for i in range(window.right_tabs.index("end"))] == ["Image Viewer", "3D Seed View", "Depth 3D View"]
        window.metric_depth_controls.inference_size_var.set("bad")
        with pytest.raises(ValueError, match="Inference size"):
            window.metric_depth_controls.config()
        event = MetricDepthProgress("running_inference", "Image 1/2 — running inference", 1, 2, 1, 0.5)
        window.metric_depth_controls.set_progress(event)
        assert window.metric_depth_controls.progress["value"] == 50
        assert "running inference" in window.metric_depth_controls.log_text.get("1.0", "end")
        window.viewer.display_mode_var.set("Metric range")
        window.viewer._render()
    finally:
        root.destroy()


def test_depth_viewer_native_failure_uses_canvas_once(monkeypatch):
    import tkinter as tk
    from map_builder.gui.depth_3d_viewer_panel import Depth3DViewerPanel

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tk display unavailable")
    try:
        root.withdraw()
        panel = Depth3DViewerPanel(root)
        panel.vtk_available = True
        panel._vtk_mode = "uninitialized"
        calls = {"native": 0, "canvas": 0}

        def native():
            calls["native"] += 1
            raise tk.TclError("vtkRenderingTk.dll missing")

        def canvas():
            calls["canvas"] += 1
            panel._renderer = object()
            panel._render_window = object()

        monkeypatch.setattr(panel, "_initialize_native_vtk", native)
        monkeypatch.setattr(panel, "_initialize_canvas_vtk", canvas)
        assert panel._ensure_vtk()
        assert panel._vtk_mode == "canvas"
        assert "off-screen canvas fallback" in panel.viewer_status_var.get()
        assert panel._ensure_vtk()
        assert calls == {"native": 1, "canvas": 1}
    finally:
        root.destroy()


def test_depth_viewer_secondary_fallback_failure_is_nonfatal(monkeypatch):
    import tkinter as tk
    from map_builder.gui.depth_3d_viewer_panel import Depth3DViewerPanel

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tk display unavailable")
    try:
        root.withdraw()
        panel = Depth3DViewerPanel(root)
        panel.vtk_available = True
        panel._vtk_mode = "uninitialized"
        monkeypatch.setattr(panel, "_initialize_native_vtk", lambda: (_ for _ in ()).throw(tk.TclError("missing bridge")))
        monkeypatch.setattr(panel, "_initialize_canvas_vtk", lambda: (_ for _ in ()).throw(RuntimeError("off-screen unavailable")))
        assert not panel._ensure_vtk()
        assert panel._vtk_mode == "unavailable"
        assert "off-screen unavailable" in panel.viewer_status_var.get()
        assert not panel._ensure_vtk()
    finally:
        root.destroy()


def test_installed_vtk_wheel_can_use_native_or_canvas_viewer():
    import tkinter as tk
    from map_builder.gui.depth_3d_viewer_panel import Depth3DViewerPanel

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tk display unavailable")
    panel = None
    try:
        root.withdraw()
        panel = Depth3DViewerPanel(root)
        if not panel.vtk_available:
            pytest.skip("VTK unavailable")
        assert panel._ensure_vtk()
        assert panel._vtk_mode in {"native", "canvas"}
        if panel._vtk_mode == "canvas":
            panel._render_window.SetSize(64, 48)
            panel._present()
            assert panel._canvas_photo is not None
    finally:
        if panel is not None and panel._render_window is not None:
            panel._render_window.Finalize()
        root.destroy()
