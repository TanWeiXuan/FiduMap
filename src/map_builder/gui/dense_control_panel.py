from __future__ import annotations
import tkinter as tk
from tkinter import ttk
from map_builder.dense_reconstruction.availability import check_dense_reconstruction_availability

class DenseControlPanel(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, **kwargs: object):
        super().__init__(master, text='Dense Reconstruction (Optional)', **kwargs)
        self.status_var=tk.StringVar()
        ttk.Label(self, textvariable=self.status_var, wraplength=340, justify='left').grid(row=0,column=0,sticky='ew',padx=6,pady=6)
        self.refresh_availability()
    def refresh_availability(self):
        res=check_dense_reconstruction_availability(); self.status_var.set(res.details)
