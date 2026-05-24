"""Main tkinter window for the fiducial map-builder workflow."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from map_builder.camera_models import load_camera_model_xml
from map_builder.detection import DetectionRunner
from map_builder.export import export_optimized_marker_map_csv
from map_builder.initialization import PnPInitializer, build_graph_from_store, initialize_seed_poses_from_store
from map_builder.initialization.diagnostics import format_graph_diagnostics
from map_builder.optimization import MapOptimizer
from map_builder.optimization.diagnostics import format_ba_summary
from map_builder.project import BAConfig, DetectorRunConfig, ImageIndexer, ProjectStore

from .control_panel import ControlPanel
from .image_list_panel import ImageListPanel
from .menu_bar import create_menu_bar
from .right_panel_tabs import RightPanelTabs


class MainWindow(ttk.Frame):
    def __init__(self, root: tk.Tk):
        super().__init__(root, padding=8)
        self.root = root
        self.store: ProjectStore | None = None
        self.project_folder: Path | None = None
        self.camera_config_path: Path | None = None
        self.camera_model: object | None = None
        self._detection_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._detection_thread: threading.Thread | None = None
        self._pnp_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._pnp_thread: threading.Thread | None = None
        self._graph_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._graph_thread: threading.Thread | None = None
        self._ba_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._ba_thread: threading.Thread | None = None

        self.root.title("Fiducial Map Builder")
        self.root.geometry("1180x760")
        create_menu_bar(root, self.open_image_folder_dialog, self.choose_camera_config_dialog, self.root.destroy)

        self.grid(row=0, column=0, sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=0, minsize=390)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.left_pane = ttk.PanedWindow(self, orient="vertical")
        self.left_pane.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self.image_list = ImageListPanel(
            self.left_pane,
            selection_changed=self.on_image_selection_changed,
            ignore_selected=self.ignore_selected,
            unignore_selected=self.unignore_selected,
        )
        self.controls = ControlPanel(
            self.left_pane,
            refresh_images=self.refresh_index,
            ignore_selected=self.ignore_selected,
            unignore_selected=self.unignore_selected,
            run_detection=self.run_detection,
            save_initialization_settings=self.save_initialization_settings,
            run_pnp_initialization=self.run_pnp_initialization,
            build_graph_and_seed=self.build_graph_and_seed,
            run_ba_refinement=self.run_ba_refinement,
            export_optimized_csv=self.export_optimized_csv,
        )
        self.left_pane.add(self.image_list, weight=3)
        self.left_pane.add(self.controls, weight=1)

        self.right_tabs = RightPanelTabs(self)
        self.right_tabs.grid(row=0, column=1, sticky="nsew")
        self.viewer = self.right_tabs.image_viewer
        self.map_3d_viewer = self.right_tabs.map_3d_viewer

        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def prompt_initial_folder(self) -> None:
        self.open_image_folder_dialog()

    def open_image_folder_dialog(self) -> None:
        folder = filedialog.askdirectory(title="Select image folder")
        if folder:
            self.load_image_folder(Path(folder))

    def choose_camera_config_dialog(self) -> None:
        initial_dir = str(self.project_folder) if self.project_folder else None
        path = filedialog.askopenfilename(
            title="Choose camera config XML",
            initialdir=initial_dir,
            filetypes=[("XML files", "*.xml"), ("All files", "*.*")],
        )
        if path:
            self.load_camera_config(Path(path))

    def load_image_folder(self, folder: Path) -> None:
        try:
            if self.store is not None:
                self.store.close()
            self.project_folder = folder.expanduser().resolve()
            self.store = ProjectStore.open(self.project_folder)
            self.camera_config_path = self.store.get_camera_config_path()
            self.controls.set_initialization_settings(
                self.store.get_marker_size_m(),
                self.store.get_configured_anchor_marker_id(),
            )
            self.controls.set_graph_status(format_graph_diagnostics(self.store.get_graph_diagnostics()))
            self.controls.set_ba_status(format_ba_summary(self.store.get_ba_summary()))
            ba_config = self.store.get_ba_config()
            self.controls.set_ba_config_values(
                ba_config.robust_loss_type,
                ba_config.robust_loss_scale_px,
                ba_config.corner_outlier_threshold_px,
                ba_config.marker_outlier_threshold_px,
                ba_config.run_outlier_second_pass,
            )
            self.map_3d_viewer.set_project(self.project_folder, self.store)
            self.refresh_index()
            self._load_default_or_existing_camera_config()
            self._update_path_display()
        except Exception as exc:
            messagebox.showerror("Open Image Folder Failed", str(exc))

    def refresh_index(self) -> None:
        if self.store is None or self.project_folder is None:
            self.controls.set_status("Choose an image folder first")
            return
        try:
            summary = ImageIndexer().index_folder(self.project_folder, self.store)
            self.refresh_image_list()
            self.controls.set_status(
                f"Indexed {summary.num_found} images: "
                f"{summary.num_new} new, {summary.num_updated} updated, "
                f"{summary.num_missing} missing"
            )
        except Exception as exc:
            messagebox.showerror("Image Indexing Failed", str(exc))

    def load_camera_config(self, path: Path) -> None:
        if self.store is None:
            messagebox.showinfo("No Project", "Choose an image folder before loading a camera config.")
            return
        try:
            self.camera_model = load_camera_model_xml(path)
            self.camera_config_path = path.expanduser().resolve()
            self.store.set_camera_config_path(self.camera_config_path)
            self._update_path_display()
            self.controls.set_status("Camera config loaded")
        except Exception as exc:
            messagebox.showerror("Camera Config Failed", str(exc))

    def refresh_image_list(self) -> None:
        if self.store is None:
            self.image_list.set_images([])
            return
        self.image_list.set_images(self.store.list_images())

    def ignore_selected(self) -> None:
        self._set_selected_ignored(True)

    def unignore_selected(self) -> None:
        self._set_selected_ignored(False)

    def _set_selected_ignored(self, ignored: bool) -> None:
        if self.store is None:
            return
        selected = self.image_list.selected_image_ids()
        for image_id in selected:
            self.store.set_image_ignored(image_id, ignored)
        self.refresh_image_list()
        verb = "Ignored" if ignored else "Unignored"
        self.controls.set_status(f"{verb} {len(selected)} image(s)")

    def on_image_selection_changed(self) -> None:
        if self.store is None or self.project_folder is None:
            return
        record = self.image_list.first_selected_record()
        if record is None:
            self.viewer.show_placeholder("No image selected")
            return
        if record.missing:
            self.viewer.show_placeholder(f"Missing image: {record.rel_path}")
            return
        detections = self.store.get_detections_for_image(record.id)
        self.viewer.show_image(record.absolute_path(self.project_folder), detections)
        self.map_3d_viewer.set_selected_image_id(record.id)

    def run_detection(self) -> None:
        if self.store is None or self.project_folder is None:
            self.controls.set_status("Choose an image folder first")
            return
        if self._detection_thread is not None and self._detection_thread.is_alive():
            return

        config = DetectorRunConfig(
            detector_type=self.controls.detector_type(),
            dictionary_name=self.controls.dictionary_name(),
            detector_params={"corner_refinement": "auto"},
        )
        self.controls.set_detection_running(True)
        self.controls.set_status("Starting detection")
        self._detection_thread = threading.Thread(target=self._run_detection_worker, args=(config,), daemon=True)
        self._detection_thread.start()
        self.root.after(100, self._poll_detection_queue)

    def _run_detection_worker(self, config: DetectorRunConfig) -> None:
        assert self.project_folder is not None
        worker_store: ProjectStore | None = None
        try:
            worker_store = ProjectStore.open(self.project_folder)

            def progress(index: int, total: int, image: object, message: str) -> None:
                self._detection_queue.put(("progress", (index, total, message)))

            runner = DetectionRunner(self.project_folder, worker_store, config, progress)
            total = runner.run()
            self._detection_queue.put(("done", total))
        except Exception as exc:
            self._detection_queue.put(("error", str(exc)))
        finally:
            if worker_store is not None:
                worker_store.close()

    def _poll_detection_queue(self) -> None:
        keep_polling = self._detection_thread is not None and self._detection_thread.is_alive()
        while True:
            try:
                kind, payload = self._detection_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "progress":
                index, total, message = payload  # type: ignore[misc]
                self.controls.set_status(f"Detection: image {index} / {total} ({message})")
            elif kind == "done":
                self.controls.set_detection_running(False)
                self.controls.set_status(f"Detection complete for {payload} image(s)")
                self.refresh_image_list()
                self.on_image_selection_changed()
            elif kind == "error":
                self.controls.set_detection_running(False)
                self.controls.set_status("Detection failed")
                messagebox.showerror("Detection Failed", str(payload))
        if keep_polling:
            self.root.after(100, self._poll_detection_queue)

    def save_initialization_settings(self) -> bool:
        if self.store is None:
            self.controls.set_pnp_status("Choose an image folder first")
            return False
        try:
            marker_size = self.controls.marker_size_m()
            anchor = self.controls.anchor_marker_id()
            self.store.set_marker_size_m(marker_size)
            self.store.set_anchor_marker_id(anchor)
            self.controls.set_initialization_settings(marker_size, anchor)
            if anchor is None:
                self.controls.set_pnp_status(
                    f"Marker geometry settings saved; auto anchor will use marker {self.store.get_anchor_marker_id()}"
                )
            else:
                self.controls.set_pnp_status("Marker geometry settings saved")
            self.map_3d_viewer.request_refresh()
            return True
        except Exception as exc:
            messagebox.showerror("Invalid Initialization Settings", str(exc))
            return False

    def run_pnp_initialization(self) -> None:
        if self.store is None or self.project_folder is None:
            self.controls.set_pnp_status("Choose an image folder first")
            return
        if self.camera_config_path is None:
            self.controls.set_pnp_status("Choose a camera config XML first")
            return
        if self._pnp_thread is not None and self._pnp_thread.is_alive():
            return
        if not self.save_initialization_settings():
            return

        marker_size = self.controls.marker_size_m()
        camera_path = self.camera_config_path
        self.controls.set_pnp_running(True)
        self.controls.set_pnp_status("Starting PnP initialization")
        self._pnp_thread = threading.Thread(
            target=self._run_pnp_worker,
            args=(camera_path, marker_size),
            daemon=True,
        )
        self._pnp_thread.start()
        self.root.after(100, self._poll_pnp_queue)

    def _run_pnp_worker(self, camera_path: Path, marker_size: float) -> None:
        assert self.project_folder is not None
        worker_store: ProjectStore | None = None
        try:
            camera_model = load_camera_model_xml(camera_path)
            worker_store = ProjectStore.open(self.project_folder)

            def progress(index: int, total: int, message: str) -> None:
                self._pnp_queue.put(("progress", (index, total, message)))

            observations = PnPInitializer(
                self.project_folder,
                worker_store,
                camera_model,
                marker_size,
                progress_callback=progress,
            ).run()
            successes = [obs for obs in observations if obs.success]
            failures = [obs for obs in observations if not obs.success]
            errors = [obs.reprojection_error_px for obs in successes if obs.reprojection_error_px is not None]
            mean_error = sum(errors) / len(errors) if errors else None
            self._pnp_queue.put(("done", (len(observations), len(successes), len(failures), mean_error)))
        except Exception as exc:
            self._pnp_queue.put(("error", str(exc)))
        finally:
            if worker_store is not None:
                worker_store.close()

    def _poll_pnp_queue(self) -> None:
        keep_polling = self._pnp_thread is not None and self._pnp_thread.is_alive()
        while True:
            try:
                kind, payload = self._pnp_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                index, total, message = payload  # type: ignore[misc]
                self.controls.set_pnp_status(f"PnP: detection {index} / {total} ({message})")
            elif kind == "done":
                total, successes, failures, mean_error = payload  # type: ignore[misc]
                self.controls.set_pnp_running(False)
                error_text = "n/a" if mean_error is None else f"{mean_error:.3f} px"
                self.controls.set_pnp_status(
                    f"Processed: {total}\nSuccessful: {successes}\nFailed: {failures}\nMean reprojection error: {error_text}"
                )
                self.map_3d_viewer.request_refresh()
            elif kind == "error":
                self.controls.set_pnp_running(False)
                self.controls.set_pnp_status("PnP initialization failed")
                messagebox.showerror("PnP Initialization Failed", str(payload))
        if keep_polling:
            self.root.after(100, self._poll_pnp_queue)

    def build_graph_and_seed(self) -> None:
        if self.store is None or self.project_folder is None:
            self.controls.set_graph_status("Choose an image folder first")
            return
        if self._graph_thread is not None and self._graph_thread.is_alive():
            return
        if not self.save_initialization_settings():
            return
        self.controls.set_graph_running(True)
        self.controls.set_graph_status("Building graph and seed poses")
        self._graph_thread = threading.Thread(target=self._run_graph_worker, daemon=True)
        self._graph_thread.start()
        self.root.after(100, self._poll_graph_queue)

    def _run_graph_worker(self) -> None:
        assert self.project_folder is not None
        worker_store: ProjectStore | None = None
        try:
            worker_store = ProjectStore.open(self.project_folder)
            build_graph_from_store(worker_store)
            initialize_seed_poses_from_store(worker_store)
            self._graph_queue.put(("done", worker_store.get_graph_diagnostics()))
        except Exception as exc:
            self._graph_queue.put(("error", str(exc)))
        finally:
            if worker_store is not None:
                worker_store.close()

    def _poll_graph_queue(self) -> None:
        keep_polling = self._graph_thread is not None and self._graph_thread.is_alive()
        while True:
            try:
                kind, payload = self._graph_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "done":
                self.controls.set_graph_running(False)
                self.controls.set_graph_status(format_graph_diagnostics(payload))  # type: ignore[arg-type]
                self.map_3d_viewer.request_refresh()
            elif kind == "error":
                self.controls.set_graph_running(False)
                self.controls.set_graph_status("Graph / seed initialization failed")
                messagebox.showerror("Graph Initialization Failed", str(payload))
        if keep_polling:
            self.root.after(100, self._poll_graph_queue)

    def run_ba_refinement(self) -> None:
        if self.store is None or self.project_folder is None:
            self.controls.set_ba_status("Choose an image folder first")
            return
        if self.camera_config_path is None:
            self.controls.set_ba_status("Choose a camera config XML first")
            return
        if self._ba_thread is not None and self._ba_thread.is_alive():
            return
        try:
            loss_type, loss_scale, corner_threshold, marker_threshold, second_pass = self.controls.ba_config_values()
            config = BAConfig(
                robust_loss_type=loss_type,
                robust_loss_scale_px=loss_scale,
                corner_outlier_threshold_px=corner_threshold,
                marker_outlier_threshold_px=marker_threshold,
                run_outlier_second_pass=second_pass,
            )
            self.store.set_ba_config(config)
        except Exception as exc:
            messagebox.showerror("Invalid BA Settings", str(exc))
            return
        self.controls.set_ba_running(True)
        self.controls.set_ba_status("Starting BA refinement")
        self._ba_thread = threading.Thread(target=self._run_ba_worker, args=(self.camera_config_path, config), daemon=True)
        self._ba_thread.start()
        self.root.after(100, self._poll_ba_queue)

    def _run_ba_worker(self, camera_path: Path, config: BAConfig) -> None:
        assert self.project_folder is not None
        worker_store: ProjectStore | None = None
        try:
            camera_model = load_camera_model_xml(camera_path)
            worker_store = ProjectStore.open(self.project_folder)

            def progress(message: str) -> None:
                self._ba_queue.put(("progress", message))

            result = MapOptimizer(worker_store, camera_model).run_ba(config, progress)
            self._ba_queue.put(("done", result.summary))
        except Exception as exc:
            self._ba_queue.put(("error", str(exc)))
        finally:
            if worker_store is not None:
                worker_store.close()

    def _poll_ba_queue(self) -> None:
        keep_polling = self._ba_thread is not None and self._ba_thread.is_alive()
        while True:
            try:
                kind, payload = self._ba_queue.get_nowait()
            except queue.Empty:
                break
            if kind == "progress":
                self.controls.set_ba_status(str(payload))
            elif kind == "done":
                self.controls.set_ba_running(False)
                self.controls.set_ba_status(format_ba_summary(payload))  # type: ignore[arg-type]
                self.map_3d_viewer.request_refresh()
            elif kind == "error":
                self.controls.set_ba_running(False)
                self.controls.set_ba_status("BA refinement failed")
                messagebox.showerror("BA Refinement Failed", str(payload))
        if keep_polling:
            self.root.after(100, self._poll_ba_queue)

    def export_optimized_csv(self) -> None:
        if self.store is None:
            self.controls.set_ba_status("Choose an image folder first")
            return
        output = filedialog.asksaveasfilename(
            title="Export optimized marker map CSV",
            defaultextension=".csv",
            initialfile="marker_map_points.csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not output:
            return
        try:
            count = export_optimized_marker_map_csv(self.store, Path(output))
            self.controls.set_ba_status(f"Exported {count} marker corner point(s) to CSV")
            messagebox.showinfo("CSV Export Complete", f"Exported {count} marker corner point(s).")
        except Exception as exc:
            messagebox.showerror("CSV Export Failed", str(exc))

    def _load_default_or_existing_camera_config(self) -> None:
        if self.store is None or self.project_folder is None:
            return
        if self.camera_config_path is not None and self.camera_config_path.exists():
            try:
                self.camera_model = load_camera_model_xml(self.camera_config_path)
                return
            except Exception:
                self.camera_config_path = None

        default_path = self.project_folder / "camera_params.xml"
        if default_path.exists():
            self.load_camera_config(default_path)
            return
        self._update_path_display()
        self.choose_camera_config_dialog()

    def _update_path_display(self) -> None:
        folder_text = str(self.project_folder) if self.project_folder else None
        camera_text = str(self.camera_config_path) if self.camera_config_path else None
        self.controls.set_paths(folder_text, camera_text)

    def close(self) -> None:
        if self.store is not None:
            self.store.close()
            self.store = None
        self.root.destroy()
