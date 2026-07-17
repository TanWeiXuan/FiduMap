def test_dense_gui_imports():
    import map_builder.gui.dense_control_panel
    import map_builder.gui.image_viewer_panel
    import map_builder.gui.map_3d_viewer_panel


def test_dense_gui_config_exposes_only_points_only_ba():
    from map_builder.gui.dense_control_panel import DenseControlPanel

    panel = DenseControlPanel.__new__(DenseControlPanel)
    assert panel.dense_ba_config().mode == "points_only"


def test_dense_count_text_and_run_all_control():
    import tkinter as tk

    import pytest

    from map_builder.gui.dense_control_panel import DenseControlPanel

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tk display unavailable")
    try:
        root.withdraw()
        panel = DenseControlPanel(
            root,
            run_extract_features=lambda: None,
            run_build_pairs=lambda: None,
            run_match_pairs=lambda: None,
            run_filter_matches=lambda: None,
            run_build_tracks=lambda: None,
            run_merge_duplicates=lambda: None,
            run_dense_ba=lambda: None,
            export_dense_csv=lambda: None,
        )
        panel.set_counts(
            {
                "feature_images": 2,
                "keypoints": 3500,
                "features": 2,
                "pairs": 3,
                "matched_pairs": 2,
                "matches": 4,
                "inliers": 2,
                "tracks": 1,
                "points": 1,
            }
        )
        text = panel.counts_var.get()
        assert "Feature images: 2" in text
        assert "Keypoints: 3,500" in text
        assert "Frame pairs: 3" in text
        assert "Matched pairs: 2" in text
        assert "Raw matches: 4" in text
        assert "Inliers: 2 (50.0%)" in text
        assert "Active points: 1 (50.0% of inliers)" in text
        assert "Features:" not in text
        assert "Matches:" not in text
        assert panel.run_all_button.cget("text") == "Run Full Dense Reconstruction"
        assert panel.button_by_stage["all"] is panel.run_all_button
    finally:
        root.destroy()


def test_dense_panel_rejects_invalid_numeric_settings():
    import tkinter as tk

    import pytest

    from map_builder.gui.dense_control_panel import DenseControlPanel

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tk display unavailable")
    try:
        root.withdraw()
        panel = DenseControlPanel(
            root,
            run_extract_features=lambda: None,
            run_build_pairs=lambda: None,
            run_match_pairs=lambda: None,
            run_filter_matches=lambda: None,
            run_build_tracks=lambda: None,
            run_merge_duplicates=lambda: None,
            run_dense_ba=lambda: None,
            export_dense_csv=lambda: None,
        )
        panel.max_pairs_var.set("0")
        with pytest.raises(ValueError, match="Max pairs / image"):
            panel.pair_selection_config()
        panel.max_pairs_var.set("10")
        panel.min_match_score_var.set("1.5")
        with pytest.raises(ValueError, match="Min descriptor score"):
            panel.matching_config()
    finally:
        root.destroy()
