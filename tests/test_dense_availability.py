from map_builder.dense_reconstruction.availability import check_dense_reconstruction_availability

def test_availability_structured():
    r=check_dense_reconstruction_availability()
    assert isinstance(r.available,bool)
    assert isinstance(r.missing_dependencies,list)
    assert isinstance(r.details,str)
