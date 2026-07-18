"""Opt-in real-model checks. These never download checkpoints."""

import os
from pathlib import Path

import numpy as np
import pytest

from map_builder.metric_depth.backends.depth_anything_v2 import DepthAnythingV2AlignedBackend
from map_builder.metric_depth.models import MetricDepthConfig


def _local_model(env_name: str) -> Path:
    value = os.environ.get(env_name)
    if not value:
        pytest.skip(f"{env_name} is not set to a local checkpoint directory")
    path = Path(value)
    if not path.is_dir():
        pytest.skip(f"{env_name} is not a valid local checkpoint directory")
    pytest.importorskip("transformers")
    return path


@pytest.mark.metric_depth_models
def test_real_dav2_tiny_image_local_checkpoint_only():
    model = _local_model("FIDUMAP_DAV2_MODEL")
    config = MetricDepthConfig(model_id_or_path=str(model), device="cpu", allow_download=False, inference_size=56)
    backend = DepthAnythingV2AlignedBackend(); backend.load(config)
    prediction = backend.predict_relative(np.zeros((32, 32, 3), dtype=np.uint8), config)
    backend.close()
    assert prediction.shape == (32, 32) and np.all(np.isfinite(prediction))
