def test_gui_modules_import_without_creating_tk_root() -> None:
    import map_builder.gui.app
    import map_builder.gui.control_panel
    import map_builder.gui.image_list_panel
    import map_builder.gui.image_viewer_panel
    import map_builder.gui.main_window
    import map_builder.gui.menu_bar
    import map_builder.gui.scrollable_frame
    import map_builder.gui.theme

    assert map_builder.gui.app.main is not None


def test_gui_app_can_load_from_script_path() -> None:
    from pathlib import Path
    import runpy

    app_path = Path(__file__).resolve().parents[1] / "src" / "map_builder" / "gui" / "app.py"
    globals_after_load = runpy.run_path(str(app_path), run_name="not_main")
    assert globals_after_load["MainWindow"].__name__ == "MainWindow"
