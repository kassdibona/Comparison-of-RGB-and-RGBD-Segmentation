"""Dummy Dataset."""

import tensorflow_datasets as tfds
from . import dummy_dataset


class DummyDatasetTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for dummy_dataset dataset."""
    # TODO:
    DATASET_CLASS = dummy_dataset.DummyDataset
    SPLITS = {
        'train': 3,  # Number of fake train example
        'test': 1,  # Number of fake test example
    }


if __name__ == '__main__':
    tfds.testing.test_main()
