def test_dense_gui_imports():
    import map_builder.gui.dense_control_panel
    import map_builder.gui.image_viewer_panel
    import map_builder.gui.map_3d_viewer_panel


def test_dense_count_text_names_feature_images_and_keypoints():
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
                "inliers": 5,
                "tracks": 6,
                "points": 7,
            }
        )
        text = panel.counts_var.get()
        assert "Feature images: 2" in text
        assert "Keypoints: 3,500" in text
        assert "Frame pairs: 3" in text
        assert "Matched pairs: 2" in text
        assert "Raw matches: 4" in text
        assert "Features:" not in text
        assert "Matches:" not in text
    finally:
        root.destroy()
