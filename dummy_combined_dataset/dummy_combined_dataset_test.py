"""Dummy Combined Dataset."""

import tensorflow_datasets as tfds
from . import dummy_combined_dataset


class DummyCombinedDatasetTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for dummy_combined_dataset dataset."""
  # TODO:
  DATASET_CLASS = dummy_combined_dataset.DummyCombinedDataset
  SPLITS = {
      'train': 3,  # Number of fake train example
      'test': 1,  # Number of fake test example
  }

if __name__ == '__main__':
  tfds.testing.test_main()
