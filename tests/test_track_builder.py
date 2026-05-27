from map_builder.dense_reconstruction.track_builder import build_tracks_union_find


def test_track_building():
    tracks = build_tracks_union_find([((0, 0), (1, 1)), ((1, 1), (2, 2))])
    assert len(tracks) == 1
    assert tracks[0] == {(0, 0), (1, 1), (2, 2)}


def test_track_building_rejects_duplicate_image_union():
    tracks = build_tracks_union_find([((0, 0), (1, 1)), ((1, 1), (2, 2)), ((2, 2), (0, 3))])
    assert len(tracks) == 1
    assert tracks[0] == {(0, 0), (1, 1), (2, 2)}
