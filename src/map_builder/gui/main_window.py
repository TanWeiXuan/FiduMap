"""Main tkinter window for the fiducial map-builder workflow."""

from __future__ import annotations

from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from map_builder.camera_models import load_camera_model_xml
from map_builder.detection import DetectionRunner
from map_builder.project import DetectorRunConfig, ImageIndexer, ProjectStore

from .control_panel import ControlPanel
from .image_list_panel import ImageListPanel
from .image_viewer_panel import ImageViewerPanel
from .menu_bar import create_menu_bar


class MainWindow(ttk.Frame):
    def __init__(self, root: tk.Tk):
        super().__init__(root, padding=8)
        self.root = root
        self.store: ProjectStore | None = None
        self.project_folder: Path | None = None
        self.camera_config_path: Path | None = None
        self._detection_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self._detection_thread: threading.Thread | None = None

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
        )
        self.left_pane.add(self.image_list, weight=3)
        self.left_pane.add(self.controls, weight=1)

        self.viewer = ImageViewerPanel(self)
        self.viewer.grid(row=0, column=1, sticky="nsew")

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
            load_camera_model_xml(path)
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

    def _load_default_or_existing_camera_config(self) -> None:
        if self.store is None or self.project_folder is None:
            return
        if self.camera_config_path is not None and self.camera_config_path.exists():
            try:
                load_camera_model_xml(self.camera_config_path)
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
