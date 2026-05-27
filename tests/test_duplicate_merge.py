from map_builder.dense_reconstruction.duplicate_merge import merge_duplicate_points
from map_builder.dense_reconstruction.models import DensePointRecord, DuplicateMergeConfig, TrackObservationRecord


def test_duplicate_merge_accepts_nearby_tracks_without_image_conflict():
    points = [
        DensePointRecord(track_id=1, x=0, y=0, z=1, num_observations=2, mean_reprojection_error_px=0.5, max_reprojection_error_px=1.0),
        DensePointRecord(track_id=2, x=0.005, y=0, z=1, num_observations=2, mean_reprojection_error_px=0.5, max_reprojection_error_px=1.0),
    ]
    obs = {
        1: [TrackObservationRecord(1, 1, 0, 10, 10)],
        2: [TrackObservationRecord(2, 2, 0, 10, 10)],
    }
    merged, accepted = merge_duplicate_points(points, obs, DuplicateMergeConfig(duplicate_merge_radius_m=0.02))
    assert accepted == 1
    assert len(merged) == 1


def test_duplicate_merge_rejects_duplicate_image_observations():
    points = [
        DensePointRecord(track_id=1, x=0, y=0, z=1, num_observations=1),
        DensePointRecord(track_id=2, x=0.005, y=0, z=1, num_observations=1),
    ]
    obs = {
        1: [TrackObservationRecord(1, 1, 0, 10, 10)],
        2: [TrackObservationRecord(2, 1, 1, 11, 10)],
    }
    merged, accepted = merge_duplicate_points(points, obs, DuplicateMergeConfig(duplicate_merge_radius_m=0.02))
    assert accepted == 0
    assert len(merged) == 2


def test_duplicate_merge_rejects_high_error_points():
    points = [
        DensePointRecord(track_id=1, x=0, y=0, z=1, mean_reprojection_error_px=0.5, max_reprojection_error_px=1.0),
        DensePointRecord(track_id=2, x=0.005, y=0, z=1, mean_reprojection_error_px=10.0, max_reprojection_error_px=12.0),
    ]
    merged, accepted = merge_duplicate_points(points, {}, DuplicateMergeConfig(duplicate_merge_radius_m=0.02))
    assert accepted == 0
    assert len(merged) == 2
