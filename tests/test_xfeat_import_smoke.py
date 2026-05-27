import pytest
import numpy as np

from map_builder.dense_reconstruction.availability import check_dense_reconstruction_availability
from map_builder.dense_reconstruction.models import XFeatExtractionConfig

def test_xfeat_optional():
    if not check_dense_reconstruction_availability().available:
        pytest.skip('optional deps unavailable')
    from map_builder.dense_reconstruction.xfeat_extractor import XFeatSemiDenseExtractor

    extractor = XFeatSemiDenseExtractor(XFeatExtractionConfig(max_keypoints=16, resize_max_side=64, device="cpu"))
    rec = extractor.extract(np.zeros((64, 64, 3), dtype=np.uint8))
    assert rec.keypoints.shape[1] == 2
    assert rec.descriptors.shape[0] == rec.keypoints.shape[0]
