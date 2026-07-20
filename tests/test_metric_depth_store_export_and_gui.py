import importlib
import json

import cv2
import numpy as np
import pytest

from map_builder.metric_depth.export import export_artifact
from map_builder.metric_depth.models import BACKEND_DAV2, MetricDepthArtifact, MetricDepthMetrics
from map_builder.metric_depth.store import MetricDepthStore


def _artifact(image_id=2, distance=2.0):
    z = np.full((3, 4), distance, dtype=np.float32)
    valid = np.ones((3, 4), dtype=bool)
    return MetricDepthArtifact(image_id, BACKEND_DAV2, 4, 3, z, z.copy(), valid, np.full_like(z, 0.75), {"test": True}, MetricDepthMetrics(anchor_pixel_count=12, anchor_spatial_coverage=0.5, valid_output_fraction=1.0, status="success"))


def test_atomic_npz_and_metadata_round_trip_and_staleness(tmp_path):
    with MetricDepthStore.open(tmp_path) as store:
        run = store.create_run(BACKEND_DAV2, "local", {}, 7, "points=0")
        path = store.save_artifact_atomic(run, _artifact())
        assert path.exists()
        loaded = store.load_artifact(run, 2)
        assert loaded is not None and np.allclose(loaded.range_m, 2.0)
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == {"z_depth_m", "range_m", "valid_mask", "confidence", "metadata_json", "metrics_json", "image_id", "backend"}
        assert store.latest_successful_record(2, 7) is not None
        assert store.latest_successful_record(2, 8) is None
        assert store.counts(8)["stale"] == 1


def test_spline_diagnostic_arrays_round_trip_and_old_standard_only_artifact_loads(tmp_path):
    artifact = _artifact()
    shape = artifact.z_depth_m.shape
    artifact.global_spline_z_depth_m = np.full(shape, 1.9, np.float32)
    artifact.spatial_log_correction = np.full(shape, 0.02, np.float32)
    artifact.alignment_extrapolation_mask = np.zeros(shape, bool)
    artifact.anchor_mask = np.eye(shape[0], shape[1], dtype=bool)
    artifact.anchor_residual_m = np.full(shape, np.nan, np.float32)
    artifact.anchor_split = np.zeros(shape, np.uint8)
    artifact.dav2_prediction = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    with MetricDepthStore.open(tmp_path) as store:
        run = store.create_run(BACKEND_DAV2, "local", {}, 7, "points=0")
        path = store.save_artifact_atomic(run, artifact)
        loaded = store.load_artifact(run, artifact.image_id)
        assert loaded is not None
        assert np.allclose(loaded.global_spline_z_depth_m, 1.9)
        assert np.array_equal(loaded.anchor_split, artifact.anchor_split)
        assert loaded.metadata["artifact_schema_version"] == 2

        with np.load(path, allow_pickle=False) as current:
            standard = {name: current[name] for name in ("z_depth_m", "range_m", "valid_mask", "confidence")}
        np.savez_compressed(path, **standard)
        legacy = store.load_artifact(run, artifact.image_id)
        assert legacy is not None and legacy.global_spline_z_depth_m is None
        assert legacy.metadata["artifact_schema_version"] == 1


def test_legacy_spline_direction_metadata_and_metric_are_ignored(tmp_path):
    artifact = _artifact()
    artifact.metadata["spline_direction"] = "decreasing"
    with MetricDepthStore.open(tmp_path) as store:
        run = store.create_run(BACKEND_DAV2, "local", {}, 7, "points=0")
        path = store.save_artifact_atomic(run, artifact)
        with np.load(path, allow_pickle=False) as current:
            values = {name: current[name] for name in current.files}
        metrics = json.loads(str(values["metrics_json"].item()))
        metrics["spline_direction"] = "decreasing"
        values["metrics_json"] = np.array(json.dumps(metrics))
        np.savez_compressed(path, **values)

        loaded = store.load_artifact(run, artifact.image_id)
        assert loaded is not None
        assert "spline_direction" not in loaded.metadata
        assert not hasattr(loaded.metrics, "spline_direction")


def test_portable_uint16_export_and_overflow_warning(tmp_path):
    artifact = _artifact(distance=2.345)
    paths = export_artifact(artifact, tmp_path)
    with np.load(paths["npz"], allow_pickle=False) as data:
        assert set(data.files) == {"z_depth_m", "range_m", "valid_mask", "confidence"}
    assert set(paths) == {"npz", "metadata", "range_mm", "confidence"}
    range_mm = cv2.imread(str(paths["range_mm"]), cv2.IMREAD_UNCHANGED)
    assert range_mm.dtype == np.uint16 and range_mm[0, 0] == 2345
    overflow = _artifact(distance=70.0)
    overflow_paths = export_artifact(overflow, tmp_path, stem="overflow")
    metadata = overflow_paths["metadata"].read_text(encoding="utf-8")
    assert '"portable_range_mm_overflow": true' in metadata
    assert cv2.imread(str(overflow_paths["range_mm"]), cv2.IMREAD_UNCHANGED)[0, 0] == 0


def test_legacy_dav2_npz_extra_arrays_are_ignored(tmp_path):
    with MetricDepthStore.open(tmp_path) as store:
        run = store.create_run(BACKEND_DAV2, "local", {}, 7, "points=0")
        path = store.save_artifact_atomic(run, _artifact())
        with np.load(path, allow_pickle=False) as current:
            values = {name: current[name] for name in current.files}
        values["prompt_depth_z_m"] = np.ones((3, 4), dtype=np.float32)
        values["prompt_mask"] = np.ones((3, 4), dtype=np.uint8)
        values["metrics_json"] = np.array('{"prompt_pixel_count": 12, "prompt_spatial_coverage": 0.5, "status": "success"}')
        np.savez_compressed(path, **values)
        loaded = store.load_artifact(run, 2)
        assert loaded is not None
        assert loaded.metrics.anchor_pixel_count == 12
        assert loaded.metrics.anchor_spatial_coverage == 0.5
        assert not hasattr(loaded, "prompt_mask")


def test_legacy_non_dav2_runs_are_preserved_but_ignored(tmp_path):
    with MetricDepthStore.open(tmp_path) as store:
        legacy = store.create_run("prompt_depth_anything", "legacy", {}, 7, "points=0")
        with store.conn:
            store.conn.execute(
                "INSERT INTO depth_map_records(run_id,image_id,status,width,height,prompt_count,prompt_coverage) VALUES(?,?,?,?,?,?,?)",
                (legacy, 2, "success", 4, 3, 12, 0.5),
            )
        assert store.get_run(legacy) is not None
        assert store.latest_run_id() is None
        assert store.latest_successful_record(2, 7) is None
        assert store.counts(7)["completed"] == 0


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


def test_gui_tab_order_numeric_validation_progress_and_placeholder(tk_root):
    from map_builder.gui.main_window import MainWindow
    from map_builder.metric_depth.models import MetricDepthProgress

    window = MainWindow(tk_root)
    assert [window.workflow_tabs.tab(i, "text") for i in range(window.workflow_tabs.index("end"))] == ["Marker BA Pipeline", "Dense Reconstruction", "Metric Depth Maps"]
    assert [window.right_tabs.tab(i, "text") for i in range(window.right_tabs.index("end"))] == ["Image Viewer", "3D Seed View", "Depth 3D View"]
    window.image_list.tree.insert("", "end", iid="1", text="one")
    window.image_list.tree.insert("", "end", iid="2", text="two")
    window.image_list._records = {1: "one", 2: "two"}
    window.image_list.tree.selection_set(("1", "2"))
    window.image_list.tree.focus("2")
    assert window.image_list.focused_selected_record() == "two"
    was_available = window.depth_3d_viewer.vtk_available
    window.depth_3d_viewer.maximum_points_var.set(0)
    window.depth_3d_viewer._request_reload()
    assert "positive integer" in window.depth_3d_viewer.viewer_status_var.get()
    assert window.depth_3d_viewer.vtk_available == was_available
    window.depth_3d_viewer.maximum_points_var.set(200000)
    window.metric_depth_controls.inference_size_var.set("bad")
    with pytest.raises(ValueError, match="Inference size"):
        window.metric_depth_controls.config()
    event = MetricDepthProgress("running_inference", "Image 1/2 — running inference", 1, 2, 1, 0.5)
    window.metric_depth_controls.set_progress(event)
    assert window.metric_depth_controls.progress["value"] == 50
    assert "running inference" in window.metric_depth_controls.log_text.get("1.0", "end")
    window.viewer.display_mode_var.set("Metric range")
    window.viewer._render()


def test_depth_viewer_native_failure_uses_canvas_once(monkeypatch, tk_root):
    import tkinter as tk
    from map_builder.gui.depth_3d_viewer_panel import Depth3DViewerPanel

    panel = Depth3DViewerPanel(tk_root)
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


def test_depth_viewer_secondary_fallback_failure_is_nonfatal(monkeypatch, tk_root):
    import tkinter as tk
    from map_builder.gui.depth_3d_viewer_panel import Depth3DViewerPanel

    panel = Depth3DViewerPanel(tk_root)
    panel.vtk_available = True
    panel._vtk_mode = "uninitialized"
    monkeypatch.setattr(panel, "_initialize_native_vtk", lambda: (_ for _ in ()).throw(tk.TclError("missing bridge")))
    monkeypatch.setattr(panel, "_initialize_canvas_vtk", lambda: (_ for _ in ()).throw(RuntimeError("off-screen unavailable")))
    assert not panel._ensure_vtk()
    assert panel._vtk_mode == "unavailable"
    assert "off-screen unavailable" in panel.viewer_status_var.get()
    assert not panel._ensure_vtk()


def test_canvas_preview_requests_are_coalesced_and_finish_full_resolution(monkeypatch, tk_root):
    from map_builder.gui.depth_3d_viewer_panel import Depth3DViewerPanel

    panel = Depth3DViewerPanel(tk_root)
    panel._vtk_mode = "canvas"
    scheduled = {}
    cancelled = []
    rendered = []

    def after(delay, callback):
        identifier = f"after-{len(scheduled)}"
        scheduled[identifier] = (delay, callback)
        return identifier

    monkeypatch.setattr(panel, "after", after)
    monkeypatch.setattr(panel, "after_cancel", cancelled.append)
    monkeypatch.setattr(panel, "_safe_present", lambda preview=False: rendered.append(preview))
    panel._request_canvas_preview()
    panel._request_canvas_preview()
    assert len(scheduled) == 1
    identifier, (delay, callback) = next(iter(scheduled.items()))
    assert delay == 33
    callback()
    assert rendered == [True]
    panel._request_canvas_full()
    full_id = panel._canvas_full_after_id
    assert scheduled[full_id][0] == 0
    scheduled[full_id][1]()
    assert rendered == [True, False]
    assert identifier not in cancelled


def test_stale_background_depth_scene_result_is_ignored(monkeypatch, tk_root):
    from concurrent.futures import Future
    from map_builder.gui.main_window import MainWindow
    from map_builder.metric_depth.depth_scene import DepthSceneResult

    window = MainWindow(tk_root)
    try:
        applied = []
        monkeypatch.setattr(window, "_apply_depth_scene_result", applied.append)
        stale = Future(); stale.set_result(DepthSceneResult(selected_count=1))
        current_result = DepthSceneResult(selected_count=2)
        current = Future(); current.set_result(current_result)
        window._depth_scene_request_id = 2
        window._depth_scene_future = current
        window._depth_scene_queue.put((1, stale))
        window._depth_scene_queue.put((2, current))
        window._poll_depth_scene_queue()
        assert applied == [current_result]
    finally:
        window._depth_scene_executor.shutdown(wait=False, cancel_futures=True)


def test_depth_scene_selection_refresh_is_debounced(monkeypatch, tmp_path, tk_root):
    from map_builder.gui.main_window import MainWindow

    window = MainWindow(tk_root)
    try:
        window.store = object()
        window.project_folder = tmp_path
        window.image_list.tree.insert("", "end", iid="1", text="one")
        window.image_list.tree.selection_set("1")
        scheduled = {}
        cancelled = []

        def after(delay, callback):
            identifier = f"request-{len(scheduled)}"
            scheduled[identifier] = (delay, callback)
            return identifier

        monkeypatch.setattr(tk_root, "after", after)
        monkeypatch.setattr(tk_root, "after_cancel", cancelled.append)
        window._schedule_depth_scene_refresh()
        first = window._depth_scene_after_id
        window._schedule_depth_scene_refresh()
        second = window._depth_scene_after_id
        assert first != second
        assert first in cancelled
        assert scheduled[second][0] == 250
    finally:
        window._depth_scene_executor.shutdown(wait=False, cancel_futures=True)


def test_installed_vtk_wheel_can_use_native_or_canvas_viewer(tk_root):
    from map_builder.camera_models import PinholeRadTanCameraModel
    from map_builder.geometry import SE3
    from map_builder.gui.depth_3d_viewer_panel import Depth3DViewerPanel
    from map_builder.metric_depth.depth_scene import DepthCloudLayer

    panel = None
    try:
        panel = Depth3DViewerPanel(tk_root)
        if not panel.vtk_available:
            pytest.skip("VTK unavailable")
        assert panel._ensure_vtk()
        assert panel._vtk_mode in {"native", "canvas"}
        camera = PinholeRadTanCameraModel(4, 3, 2, 2, 1.5, 1, 0, 0, 0, 0, 0)
        layers = [
            DepthCloudLayer(
                image_id=index,
                source_run_id=1,
                points_world=np.array([[index, 0, 2], [index, 1, 2]], dtype=np.float32),
                rgb=np.full((2, 3), 100 + index, dtype=np.uint8),
                range_m=np.full(2, 2.0, dtype=np.float32),
                confidence=np.ones(2, dtype=np.float32),
                T_W_C=SE3(np.eye(3), np.array([float(index), 0.0, 0.0])),
                candidate_count=2,
                sampled_candidate_count=2,
            )
            for index in (1, 2)
        ]
        panel.set_depth_clouds(layers, camera, selected_count=2, skipped_count=0)
        panel.refresh()
        point_actor = panel._point_actor
        assert point_actor.GetMapper().GetInput().GetNumberOfPoints() == 4
        assert panel._frustum_actor.GetMapper().GetInput().GetNumberOfLines() == 16
        panel.color_mode_var.set("Confidence")
        panel.point_size_var.set(3.0)
        panel._update_presentation()
        assert panel._point_actor is point_actor
        assert point_actor.GetProperty().GetPointSize() == 3.0
        if panel._vtk_mode == "canvas":
            panel._canvas_requested_size = (64, 48)
            capture = panel._capture_filter
            panel._present(preview=True)
            assert panel._render_window.GetSize() == (32, 24)
            assert panel._capture_filter is capture
            panel._present()
            assert panel._render_window.GetSize() == (64, 48)
            assert panel._canvas_photo is not None
        panel.clear_metric_depth()
        assert panel._renderer.GetViewProps().GetNumberOfItems() == 0
    finally:
        if panel is not None and panel._render_window is not None:
            panel._render_window.Finalize()
