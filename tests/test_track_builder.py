import numpy as np

from map_builder.dense_reconstruction.models import (
    FramePairRecord,
    PairMatchRecord,
    TriangulationConfig,
)
from map_builder.dense_reconstruction import track_builder
from map_builder.dense_reconstruction.track_builder import build_tracks_union_find


def test_track_building():
    tracks = build_tracks_union_find([((0, 0), (1, 1)), ((1, 1), (2, 2))])
    assert len(tracks) == 1
    assert tracks[0] == {(0, 0), (1, 1), (2, 2)}


def test_track_building_rejects_duplicate_image_union():
    tracks = build_tracks_union_find([((0, 0), (1, 1)), ((1, 1), (2, 2)), ((2, 2), (0, 3))])
    assert len(tracks) == 1
    assert tracks[0] == {(0, 0), (1, 1), (2, 2)}


def test_track_builder_prioritizes_strong_edge_before_conflicting_weak_edge(monkeypatch):
    pairs = [
        FramePairRecord(id=1, image_id_a=0, image_id_b=1),
        FramePairRecord(id=2, image_id_a=1, image_id_b=2),
        FramePairRecord(id=3, image_id_a=0, image_id_b=2),
    ]
    strong_01 = PairMatchRecord(1, 0, 0, 10, 10, 11, 10, match_score=0.9, epipolar_error=0.001, is_epipolar_inlier=1)
    strong_12 = PairMatchRecord(2, 0, 0, 11, 10, 12, 10, match_score=0.9, epipolar_error=0.001, is_epipolar_inlier=1)
    weak_conflict = PairMatchRecord(3, 1, 0, 20, 20, 12, 10, match_score=0.1, epipolar_error=0.1, is_epipolar_inlier=1)

    def fake_triangulate(observations, _poses, _camera, _config):
        return (
            np.array([0.0, 0.0, 5.0]),
            {
                "mean_reprojection_error_px": 0.1,
                "max_reprojection_error_px": 0.2,
                "min_triangulation_angle_deg": 2.0,
            },
            np.ones((len(observations),), dtype=bool),
        )

    monkeypatch.setattr(track_builder, "triangulate_multiview_robust", fake_triangulate)
    results = track_builder.build_tracks_from_matches(
        pairs,
        {3: [weak_conflict], 1: [strong_01], 2: [strong_12]},
        poses_by_image={},
        camera_model=object(),
        config=TriangulationConfig(),
    )

    assert len(results) == 1
    track, observations, _point = results[0]
    assert track.num_images == 3
    assert {(obs.image_id, obs.feature_idx) for obs in observations} == {(0, 0), (1, 0), (2, 0)}
