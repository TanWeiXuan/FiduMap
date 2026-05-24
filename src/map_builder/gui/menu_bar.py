"""Application menu bar."""

from __future__ import annotations

import tkinter as tk


def create_menu_bar(
    root: tk.Tk,
    open_image_folder: object,
    choose_camera_config: object,
    exit_app: object,
) -> tk.Menu:
    menu_bar = tk.Menu(root)
    file_menu = tk.Menu(menu_bar, tearoff=False)
    file_menu.add_command(label="Open Image Folder", command=open_image_folder)
    file_menu.add_command(label="Choose Camera Config XML", command=choose_camera_config)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=exit_app)
    menu_bar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menu_bar)
    return menu_bar
