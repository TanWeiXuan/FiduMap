import pytest
from map_builder.dense_reconstruction.availability import check_dense_reconstruction_availability

def test_xfeat_optional():
    if not check_dense_reconstruction_availability().available:
        pytest.skip('optional deps unavailable')
