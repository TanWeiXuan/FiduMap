"""Image table panel."""

from __future__ import annotations

from collections.abc import Callable
import tkinter as tk
from tkinter import ttk

from map_builder.project.models import ImageRecord


FILLED = "\u25cf"
EMPTY = "\u25cb"


class ImageListPanel(ttk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        selection_changed: Callable[[], None],
        ignore_selected: Callable[[], None],
        unignore_selected: Callable[[], None],
        **kwargs: object,
    ):
        super().__init__(master, **kwargs)
        self._records: dict[int, ImageRecord] = {}
        self.tree = ttk.Treeview(
            self,
            columns=("ignored", "detected", "stale", "markers"),
            show="tree headings",
            selectmode="extended",
        )
        self.tree.heading("#0", text="Image")
        self.tree.heading("ignored", text="Ignored")
        self.tree.heading("detected", text="Detected")
        self.tree.heading("stale", text="Stale")
        self.tree.heading("markers", text="Markers")
        self.tree.column("#0", width=260, minwidth=160)
        self.tree.column("ignored", width=70, anchor="center", stretch=False)
        self.tree.column("detected", width=75, anchor="center", stretch=False)
        self.tree.column("stale", width=60, anchor="center", stretch=False)
        self.tree.column("markers", width=70, anchor="e", stretch=False)

        y_scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        x_scroll = ttk.Scrollbar(self, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.tree.bind("<<TreeviewSelect>>", lambda _event: selection_changed())
        self.tree.bind("<Button-3>", self._show_context_menu)
        self._context_menu = tk.Menu(self, tearoff=False)
        self._context_menu.add_command(label="Ignore selected", command=ignore_selected)
        self._context_menu.add_command(label="Unignore selected", command=unignore_selected)

    def set_images(self, images: list[ImageRecord]) -> None:
        current_selection = set(self.selected_image_ids())
        self.tree.delete(*self.tree.get_children())
        self._records = {image.id: image for image in images}
        for image in images:
            detected = image.detection_status in {"detected", "no_markers"} and not image.modified_since_detection
            stale = image.modified_since_detection or image.detection_status == "stale"
            display_path = f"{image.rel_path} (missing)" if image.missing else image.rel_path
            self.tree.insert(
                "",
                "end",
                iid=str(image.id),
                text=display_path,
                values=(
                    FILLED if image.ignored else EMPTY,
                    FILLED if detected else EMPTY,
                    FILLED if stale else EMPTY,
                    image.marker_count,
                ),
            )
        for image_id in current_selection:
            if image_id in self._records:
                self.tree.selection_add(str(image_id))

    def selected_image_ids(self) -> list[int]:
        return [int(iid) for iid in self.tree.selection()]

    def first_selected_record(self) -> ImageRecord | None:
        image_ids = self.selected_image_ids()
        if not image_ids:
            return None
        return self._records.get(image_ids[0])

    def _show_context_menu(self, event: tk.Event) -> None:
        row = self.tree.identify_row(event.y)
        if row and row not in self.tree.selection():
            self.tree.selection_set(row)
        self._context_menu.tk_popup(event.x_root, event.y_root)
