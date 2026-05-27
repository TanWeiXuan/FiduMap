from map_builder.dense_reconstruction.track_builder import build_tracks_union_find

def test_track_building():
    tracks=build_tracks_union_find([((0,0),(1,1)),((1,1),(2,2))])
    assert len(tracks)==1 and len(tracks[0])==3
