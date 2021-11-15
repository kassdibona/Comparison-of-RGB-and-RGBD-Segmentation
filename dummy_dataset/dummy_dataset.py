# Based upon 
# https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/oxford_iiit_pet.py which is

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# NOTE
# trainval.txt and test.txt should be within the BASE_PATH
# folders `images`, `depths`, and `ground` should also be within the BASE_PATH

"""Dummy Dataset."""

import os
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# Markdown description that will appear on the catalog page and in the info.
# TODO: Change this description per dataset
_DESCRIPTION = """
This Dummy Dataset has n categories of images within the dataset that has a total of N images.
"""

# TODO: Set homepage that describing this dataset
_HOMEPAGE = 'https://dataset-homepage/'

# BibTeX citation
_CITATION = """
"""

# TODO: Change these paths per dataset
_BASE_PATH = "../data_dummy_examples"
_IMAGE_PATH = "images"
_DEPTH_PATH = "depths"
_GROUND_PATH = "ground"

_TRAIN_SPLIT_FILE = "trainval.txt"
_TEST_SPLIT_FILE = "test.txt"

_LABEL_CLASSES = ["road"]


class DummyDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dummy_dataset dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Initial release.',
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        # Specify the tfds.core.DatasetInfo object
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                # Specify the features of the dataset
                # e.g. images, labels ...
                "image":
                    tfds.features.Image(shape=(None, None, 3)),
                "depth":
                    tfds.features.Image(shape=(None, None, 3)),
                "label":
                    tfds.features.ClassLabel(names=_LABEL_CLASSES),
                "file_name":
                    tfds.features.Text(),
                "segmentation_mask":
                    tfds.features.Image(shape=(None, None, 1), use_colormap=True)
            }),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=('image', 'label'),  # Set to `None` to disable
            homepage=_HOMEPAGE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits...
        # This version assumes that the image files are stored locally
        # i.e. on the same computer that this dataset will be built
        images_path_dir = os.path.join(_BASE_PATH, _IMAGE_PATH)
        depths_path_dir = os.path.join(_BASE_PATH, _DEPTH_PATH)
        annotations_path_dir = os.path.join(_BASE_PATH, _GROUND_PATH)

        train_split_path = os.path.join(_BASE_PATH, _TRAIN_SPLIT_FILE)
        test_split_path = os.path.join(_BASE_PATH, _TEST_SPLIT_FILE)

        # Setup train and test splits
        train_split = tfds.core.SplitGenerator(
            name="train",
            gen_kwargs={
                "images_dir_path":
                    images_path_dir,
                "depths_dir_path":
                    depths_path_dir,
                "annotations_dir_path":
                    annotations_path_dir,
                "images_list_file":
                    train_split_path,
            },
        )
        test_split = tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "images_dir_path":
                    images_path_dir,
                "depths_dir_path":
                    depths_path_dir,
                "annotations_dir_path":
                    annotations_path_dir,
                "images_list_file":
                    test_split_path,
            },
        )

        return [train_split, test_split]
   

    def _generate_examples(self, images_dir_path, depths_dir_path, \
                           annotations_dir_path, images_list_file):
        """Yields examples."""
        # Yield (key, example) tuples from the dataset
        with tf.io.gfile.GFile(images_list_file, "r") as images_list:
            for line in images_list:
                init_image_name, label = line.strip().split(" ")

                ground_name = init_image_name + "-ground.png"
                depth_name  = init_image_name + "-depth.png"
                image_name  = init_image_name + "-left.png"
                label = int(label) - 1

                record = {
                    "image":
                        os.path.join(images_dir_path, image_name),
                    "depth":
                        os.path.join(depths_dir_path, depth_name),
                    "label":
                        int(label),
                    "file_name":
                        image_name,
                    "segmentation_mask":
                        os.path.join(annotations_dir_path, ground_name)
                }
                yield image_name, record

