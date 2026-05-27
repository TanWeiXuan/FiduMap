import numpy as np

from map_builder.dense_reconstruction.dense_store import (
    DenseReconstructionStore,
    numpy_array_from_blob,
    numpy_array_to_blob,
    numpy_arrays_from_blob,
    numpy_arrays_to_blob,
)
from map_builder.dense_reconstruction.models import (
    DensePointRecord,
    FramePairRecord,
    PairMatchRecord,
    TrackObservationRecord,
    TrackRecord,
)


def test_blob_roundtrip():
    a = np.array([[1, 2], [3, 4]], dtype=np.float32)
    assert np.allclose(numpy_array_from_blob(numpy_array_to_blob(a)), a)
    arrays = numpy_arrays_from_blob(numpy_arrays_to_blob({"a": a, "b": a + 1}))
    assert np.allclose(arrays["a"], a)
    assert np.allclose(arrays["b"], a + 1)


def test_store_roundtrips_dense_entities(tmp_path):
    s = DenseReconstructionStore.open(tmp_path)
    keypoints = np.array([[10, 20], [30, 40]], dtype=np.float32)
    descriptors = np.ones((2, 64), dtype=np.float32)
    scores = np.array([0.8, 0.9], dtype=np.float32)
    s.upsert_feature(1, "a.jpg", keypoints, descriptors, scores, width=100, height=80)
    feature = s.get_feature(1)
    assert feature is not None
    assert np.allclose(feature.keypoints, keypoints)
    assert np.allclose(feature.descriptors, descriptors)
    assert np.allclose(feature.scores, scores)

    pair_id = s.upsert_frame_pair(FramePairRecord(image_id_a=1, image_id_b=2, baseline_m=0.1))
    assert s.list_frame_pairs()[0].id == pair_id
    inserted = s.replace_pair_matches(
        pair_id,
        [PairMatchRecord(pair_id, 0, 1, 10, 20, 11, 21, match_score=0.5)],
    )
    assert inserted[0].id is not None
    assert inserted[0].pair_id == pair_id
    matches = s.list_pair_matches(pair_id)
    assert len(matches) == 1
    s.update_pair_epipolar_results(pair_id, [matches[0].id], np.array([0.001]), np.array([True]), min_inliers=1)
    assert s.list_pair_matches(pair_id)[0].is_epipolar_inlier == 1

    track = TrackRecord(status="active", num_observations=2, num_images=2, x=1, y=2, z=3)
    obs = [
        TrackObservationRecord(0, 1, 0, 10, 20),
        TrackObservationRecord(0, 2, 1, 11, 21),
    ]
    point = DensePointRecord(x=1, y=2, z=3, num_observations=2)
    s.replace_tracks_and_points([(track, obs, point)])
    assert len(s.list_tracks()) == 1
    assert len(s.list_track_observations()) == 2
    assert len(s.list_active_dense_points()) == 1
    run = s.create_dense_ba_run("points_only")
    s.complete_dense_ba_run(run, True, num_points=1, num_observations=2)
    counts = s.dense_counts()
    assert counts["features"] == 1
    assert counts["pairs"] == 1
    assert counts["matches"] == 1
    assert counts["inliers"] == 1
    assert counts["tracks"] == 1
    assert counts["points"] == 1
    s.close()

    s2 = DenseReconstructionStore.open(tmp_path)
    assert np.allclose(s2.get_feature(1).keypoints, keypoints)
    assert len(s2.list_active_dense_points()) == 1
