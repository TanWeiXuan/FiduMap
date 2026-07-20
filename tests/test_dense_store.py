import numpy as np

from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.models import (
    DensePointRecord,
    FramePairRecord,
    PairMatchRecord,
    TrackObservationRecord,
    TrackRecord,
)


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


def test_dense_counts_separate_feature_images_from_keypoints(tmp_path):
    store = DenseReconstructionStore.open(tmp_path)
    store.upsert_feature(
        1,
        "a.jpg",
        np.zeros((100, 2), dtype=np.float32),
        np.zeros((100, 64), dtype=np.float32),
        status="success",
    )
    store.upsert_feature(
        2,
        "b.jpg",
        np.zeros((250, 2), dtype=np.float32),
        np.zeros((250, 64), dtype=np.float32),
        status="success",
    )
    store.upsert_feature(
        3,
        "c.jpg",
        np.zeros((500, 2), dtype=np.float32),
        np.zeros((500, 64), dtype=np.float32),
        status="failed",
    )

    counts = store.dense_counts()
    assert counts["feature_images"] == 2
    assert counts["keypoints"] == 350
    assert counts["features"] == 2


def _track_result(x=1.0, feature_offset=0):
    track = TrackRecord(status="active", num_observations=2, num_images=2, x=x, y=2, z=3)
    observations = [
        TrackObservationRecord(0, 1, feature_offset, 10, 20),
        TrackObservationRecord(0, 2, feature_offset + 1, 11, 21),
    ]
    point = DensePointRecord(x=x, y=2, z=3, num_observations=2, source="triangulated")
    return track, observations, point


def test_retriangulation_removes_prior_dense_ba_and_merged_points(tmp_path):
    store = DenseReconstructionStore.open(tmp_path)
    store.replace_tracks_and_points([_track_result(x=1.0)])
    original = store.list_active_dense_points()[0]
    store.update_dense_point_coordinates(int(original["id"]), np.array([1.1, 2.0, 3.0]), source="dense_ba")
    store.replace_active_dense_points(
        [DensePointRecord(track_id=int(original["track_id"]), x=1.2, y=2, z=3)],
        source="merged",
    )
    assert {str(row["source"]) for row in store.conn.execute("SELECT source FROM dense_points")} == {
        "dense_ba",
        "merged",
    }

    store.replace_tracks_and_points([_track_result(x=9.0, feature_offset=10)])

    rows = store.conn.execute("SELECT * FROM dense_points").fetchall()
    assert len(rows) == 1
    assert rows[0]["source"] == "triangulated"
    assert rows[0]["x"] == 9.0
    orphan_count = store.conn.execute(
        """
        SELECT COUNT(*) c FROM dense_points p
        LEFT JOIN tracks t ON t.id=p.track_id
        WHERE p.is_active=1 AND p.track_id IS NOT NULL AND t.id IS NULL
        """
    ).fetchone()["c"]
    assert orphan_count == 0


def test_replace_frame_pairs_removes_complete_old_reconstruction_and_is_idempotent(tmp_path):
    store = DenseReconstructionStore.open(tmp_path)
    old_pairs = store.replace_frame_pairs(
        [
            FramePairRecord(image_id_a=1, image_id_b=2),
            FramePairRecord(image_id_a=2, image_id_b=3),
        ]
    )
    for pair in old_pairs:
        assert pair.id is not None
        store.replace_pair_matches(
            pair.id,
            [PairMatchRecord(pair.id, 0, 1, 10, 20, 11, 21, is_epipolar_inlier=1, is_used_for_track=1)],
        )
    store.replace_tracks_and_points([_track_result()])

    replacement = [FramePairRecord(image_id_a=1, image_id_b=3)]
    store.replace_frame_pairs(replacement)

    assert [(pair.image_id_a, pair.image_id_b) for pair in store.list_frame_pairs()] == [(1, 3)]
    assert store.list_pair_matches() == []
    assert store.list_tracks() == []
    assert store.list_track_observations() == []
    assert store.conn.execute("SELECT COUNT(*) c FROM dense_points").fetchone()["c"] == 0
    first_counts = store.dense_counts()

    store.replace_frame_pairs([FramePairRecord(image_id_a=1, image_id_b=3)])

    assert store.dense_counts() == first_counts
    assert [(pair.image_id_a, pair.image_id_b) for pair in store.list_frame_pairs()] == [(1, 3)]
