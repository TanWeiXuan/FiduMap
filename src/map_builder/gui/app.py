"""Tkinter entrypoint for the fiducial map builder."""

from __future__ import annotations

from pathlib import Path
import sys
import tkinter as tk

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from map_builder.gui.main_window import MainWindow
    from map_builder.gui.theme import load_azure_theme
else:
    from .main_window import MainWindow
    from .theme import load_azure_theme


def main() -> None:
    root = tk.Tk()
    load_azure_theme(root, mode="light")
    window = MainWindow(root)
    root.after(100, window.prompt_initial_folder)
    root.mainloop()


if __name__ == "__main__":
    main()
