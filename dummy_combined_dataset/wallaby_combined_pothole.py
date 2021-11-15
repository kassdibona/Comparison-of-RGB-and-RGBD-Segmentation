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

"""Dummy Combined Dataset."""

import os
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

# Markdown description that will appear on the catalog page and in the info.
# TODO: Change this description per dataset
_DESCRIPTION = """
The Dummy Combined Dataset is the combined dataset of the Dummy Dataset 1 and the Dummy Dataset 2.
"""

# TODO: Set homepage that describing this dataset
_HOMEPAGE = 'https://dataset-homepage/'

# BibTeX citation
_CITATION = """
"""

# TODO: Change these paths per dataset
_BASE_PATH = ["../data_dummy_examples1", "../data_dummy_examples2"]
_IMAGE_PATH = "images"
_DEPTH_PATH = "depths"
_GROUND_PATH = "ground"

_TRAIN_SPLIT_FILE = "trainval.txt"
_TEST_SPLIT_FILE = "test.txt"

_LABEL_CLASSES = ["pothole"]


class DummyCombinedDataset(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dummy_combined_dataset dataset."""

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
                # Specify the features of your dataset
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
            homepage='https://dataset-homepage/',
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # Downloads the data and defines the splits...
        # This version assumes that the image files are stored locally
        # i.e. on the same computer that this dataset will be built
        images_path_dirs = []
        depths_path_dirs = []
        annotations_path_dirs = []
        train_image_lists = []
        test_image_lists = []

        for _base_path in _BASE_PATH:
            images_path_dir = os.path.join(_base_path, _IMAGE_PATH)
            depths_path_dir = os.path.join(_base_path, _DEPTH_PATH)
            annotations_path_dir = os.path.join(_base_path, _GROUND_PATH)
            train_image_list = os.path.join(_base_path, _TRAIN_SPLIT_FILE)
            test_image_list = os.path.join(_base_path, _TEST_SPLIT_FILE)
            
            images_path_dirs.append(images_path_dir)
            depths_path_dirs.append(depths_path_dir)
            annotations_path_dirs.append(annotations_path_dir)
            train_image_lists.append(train_image_list)
            test_image_lists.append(test_image_list)


        # Return the Dict[split names, Iterator[Key, Example]]

        # Setup train and test splits
        train_split = tfds.core.SplitGenerator(
            name="train",
            gen_kwargs={
                "images_dir_paths":
                    images_path_dirs,
                "depths_dir_paths":
                    depths_path_dirs,
                "annotations_dir_paths":
                    annotations_path_dirs,
                "images_list_files":
                    train_image_lists,
            },
        )
        test_split = tfds.core.SplitGenerator(
            name="test",
            gen_kwargs={
                "images_dir_paths":
                    images_path_dirs,
                "depths_dir_paths":
                    depths_path_dirs,
                "annotations_dir_paths":
                    annotations_path_dirs,
                "images_list_files":
                    test_image_lists,
            },
        )

        return [train_split, test_split]


    def _generate_examples(self, images_dir_paths, depths_dir_paths, \
                           annotations_dir_paths, images_list_files):
        """Yields examples."""
        # Yield (key, example) tuples from the dataset
        for i, images_list_file in enumerate(images_list_files):
            with tf.io.gfile.GFile(images_list_file, "r") as images_list:
                for line in images_list:
                    init_image_name, label = line.strip().split(" ")

                    ground_name = init_image_name + "-ground.png"
                    depth_name  = init_image_name + "-depth.png"
                    image_name  = init_image_name + "-left.png"
                    label = int(label) - 1

                    record = {
                        "image":
                            os.path.join(images_dir_paths[i], image_name),
                        "depth":
                            os.path.join(depths_dir_paths[i], depth_name),
                        "label":
                            int(label),
                        "file_name":
                            image_name,
                        "segmentation_mask":
                            os.path.join(annotations_dir_paths[i], ground_name)
                    }
                    yield image_name, record

