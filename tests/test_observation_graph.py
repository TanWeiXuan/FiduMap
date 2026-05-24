from map_builder.geometry import SE3
from map_builder.initialization import build_observation_graph
from map_builder.project import PnPObservation


def test_observation_graph_connects_markers_through_shared_images() -> None:
    T = SE3.identity().to_json_dict()
    observations = [
        PnPObservation(image_id=0, detection_id=0, marker_id=0, success=True, T_C_M=T, id=10),
        PnPObservation(image_id=0, detection_id=1, marker_id=1, success=True, T_C_M=T, id=11),
        PnPObservation(image_id=1, detection_id=2, marker_id=1, success=True, T_C_M=T, id=12),
        PnPObservation(image_id=1, detection_id=3, marker_id=2, success=True, T_C_M=T, id=13),
    ]
    graph = build_observation_graph(observations, anchor_marker_id=0)
    assert len(graph.camera_to_edges[0]) == 2
    assert len(graph.marker_to_edges[1]) == 2
    assert (0, 1) in graph.marker_overlap_edges
    assert (1, 2) in graph.marker_overlap_edges
    assert graph.summary is not None
    assert graph.summary.markers_connected_to_anchor == 3
    assert graph.summary.disconnected_markers == []
