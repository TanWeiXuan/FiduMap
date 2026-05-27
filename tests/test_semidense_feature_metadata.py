import numpy as np

from map_builder.dense_reconstruction.dense_store import DenseReconstructionStore
from map_builder.dense_reconstruction.models import DenseFeatureRecord


def test_semidense_feature_metadata_persists(tmp_path):
    store = DenseReconstructionStore.open(tmp_path)
    store.upsert_feature_record(
        DenseFeatureRecord(
            image_id=1,
            rel_path="a.jpg",
            width=20,
            height=10,
            keypoints=np.array([[1.0, 2.0]], dtype=np.float32),
            descriptors=np.ones((1, 64), dtype=np.float32),
            scores=np.array([1.0], dtype=np.float32),
            status="success",
            num_keypoints=1,
            extraction_mode="semi_dense_xfeat",
            descriptor_source="detectAndComputeDense",
        )
    )
    store.close()

    reopened = DenseReconstructionStore.open(tmp_path)
    feature = reopened.get_feature(1)
    assert feature is not None
    assert feature.extraction_mode == "semi_dense_xfeat"
    assert feature.descriptor_source == "detectAndComputeDense"
    reopened.close()


def test_missing_semidense_metadata_loads_as_unknown(tmp_path):
    store = DenseReconstructionStore.open(tmp_path)
    store.conn.execute(
        """
        INSERT INTO dense_images(
            image_id, rel_path, features_status, num_keypoints, feature_blob, descriptor_blob
        )
        VALUES(1, 'old.jpg', 'success', 0, NULL, NULL)
        """
    )
    store.conn.commit()

    feature = store.get_feature(1)
    assert feature is not None
    assert feature.extraction_mode == "unknown"
    assert feature.descriptor_source == "unknown"
    store.close()
