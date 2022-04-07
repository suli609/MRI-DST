# MRI-DST
This project consists of two parts, Mask R-CNN based model and C3D-based-model. Mask R-CNN based model is used to segment the image, and the segmented image is input to C3D-based-model to predict the probability of ≥3 linear stapler cartridges.
To protect privacy, this project does not provide original data, only trained models and prediction codes are available.

# Mask R-CNN based model
The code of Mask R-CNN is implemented based on mmdetection (https://github.com/open-mmlab/mmdetection) with slight modifications


# C3D-based-model
## install 

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

## Test the trained model
The model(c3d-based-model.hdf5) can be downloaded at the release of this project. Then Run:

`python test.py`
