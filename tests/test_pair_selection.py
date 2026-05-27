from map_builder.dense_reconstruction.models import PairSelectionConfig
from map_builder.dense_reconstruction.pair_selection import select_frame_pairs
from map_builder.project.models import MarkerDetection, OptimizedCameraPose


def _pose(image_id, x):
    return OptimizedCameraPose(image_id=image_id, ba_run_id=1, T_W_C={"R": [[1, 0, 0], [0, 1, 0], [0, 0, 1]], "t": [x, 0, 0]})


def test_pair_selection_baseline_and_caps():
    poses = [_pose(1, 0.0), _pose(2, 0.1), _pose(3, 3.0)]
    det = MarkerDetection("aruco", "d", 10, [[0, 0], [1, 0], [1, 1], [0, 1]], "none")
    detections = {1: [det], 2: [det], 3: []}
    cfg = PairSelectionConfig(max_pairs_per_image=1, max_baseline_m=1.0)
    pairs = select_frame_pairs(poses, detections, cfg)
    assert len(pairs) == 1
    assert (pairs[0].image_id_a, pairs[0].image_id_b) == (1, 2)
