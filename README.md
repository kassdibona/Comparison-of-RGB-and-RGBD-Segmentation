# Comparison-of-RGB-and-RGBD-Segmentation
Comparison of RGB and RGB-D Segmentation with regards to Roadway (road) and Pothole Recognition

Directory data\_preprocessing contains the notebooks used to help preprocess the data prior to and as part of the dataset construction.

Directory dummy\_combined\_dataset contains the base template files used to create the local TensorFlow DataSets (TFDS) that combines two other local TFDS datasets.

Directory dummy\_dataset contains the base template files used to create a local TensorFlow DataSets (TFDS).

Directory project\_script contains the notebook and additional model definition scripts that was used to collect the results for the research these scripts were produced for.

Script svo\_save\_frame.py was produced to save frames from .svo video recording files with the flexibility of specifying the starting frame from when to start saving frames, to save every n frames, and how many total frames should be saved from the recording. It is assumed that the .svo is saved in a folder/directory with the same name (i.e. SVO\_Folder.svo should be saved in directory SVO\_Folder).
