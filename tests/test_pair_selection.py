from map_builder.dense_reconstruction.models import PairSelectionConfig
from map_builder.dense_reconstruction.pair_selection import select_frame_pairs
from map_builder.project.models import MarkerDetection, OptimizedCameraPose


def _pose(image_id, x):
    return OptimizedCameraPose(image_id=image_id, ba_run_id=1, T_W_C={"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [x, 0, 0]})


def _detection(marker_id):
    return MarkerDetection("aruco", "d", marker_id, [[0, 0], [1, 0], [1, 1], [0, 1]], "none")


def test_pair_selection_baseline_and_caps():
    poses = [_pose(1, 0.0), _pose(2, 0.1), _pose(3, 3.0)]
    det = _detection(10)
    detections = {1: [det], 2: [det], 3: []}
    cfg = PairSelectionConfig(max_pairs_per_image=1, max_baseline_m=1.0)
    pairs = select_frame_pairs(poses, detections, cfg)
    assert len(pairs) == 1
    assert (pairs[0].image_id_a, pairs[0].image_id_b) == (1, 2)


def test_pair_selection_prioritizes_marker_overlap_over_tiny_baseline():
    poses = [_pose(1, 0.0), _pose(2, 0.05), _pose(3, 0.30)]
    detections = {
        1: [_detection(10)],
        2: [_detection(20)],
        3: [_detection(10)],
    }
    cfg = PairSelectionConfig(max_pairs_per_image=1, max_baseline_m=1.0, min_common_markers=0)

    pairs = select_frame_pairs(poses, detections, cfg)

    assert [(pair.image_id_a, pair.image_id_b) for pair in pairs] == [(1, 3)]
    assert pairs[0].common_marker_count == 1


def test_pair_selection_enforces_global_degree_cap():
    poses = [_pose(1, 0.0), _pose(2, 0.2), _pose(3, 0.4), _pose(4, 0.6)]
    detections = {pose.image_id: [_detection(10)] for pose in poses}
    cfg = PairSelectionConfig(max_pairs_per_image=1, max_baseline_m=1.0)

    pairs = select_frame_pairs(poses, detections, cfg)

    degree = {pose.image_id: 0 for pose in poses}
    for pair in pairs:
        degree[pair.image_id_a] += 1
        degree[pair.image_id_b] += 1
    assert all(value <= 1 for value in degree.values())
