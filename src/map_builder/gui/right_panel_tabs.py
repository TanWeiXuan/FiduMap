"""Right-side notebook containing the image and 3D viewers."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .image_viewer_panel import ImageViewerPanel
from .map_3d_viewer_panel import Map3DViewerPanel


class RightPanelTabs(ttk.Notebook):
    def __init__(self, master: tk.Misc, **kwargs: object):
        super().__init__(master, **kwargs)
        self.image_viewer = ImageViewerPanel(self)
        self.map_3d_viewer = Map3DViewerPanel(self)
        self.add(self.image_viewer, text="Image Viewer")
        self.add(self.map_3d_viewer, text="3D Seed View")
