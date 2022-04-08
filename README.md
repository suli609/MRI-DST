# MRI-DST
This project consists of two parts, Mask R-CNN based model and C3D-based-model. Mask R-CNN based model is used to segment the image, and the segmented image is input to C3D-based-model to predict the probability of ≥3 linear stapler cartridges.
To protect privacy, this project does not provide original data, only trained models and prediction codes are available.

# Mask R-CNN based model
The code of Mask R-CNN is implemented based on mmdetection (https://github.com/open-mmlab/mmdetection) with slight modifications. 
## Install
a. Install a virtual environment.

  `conda create -n MRI python=3.7 -y`
  
  `conda activate MRI`
  
b. Install pytorch and torchversion.

  `conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch`
  
c. Install mmcv.

  `pip install mmcv-full==1.1.5 -f https://download.openmmlab.com/mmcv/dist/cu101/torch1.6.0/index.html`
  
d. Install the remaining required environment packages.

  `pip install -r requirements/build.txt`
  
  `python setup.py develop`

## Test the trained model

The model(mask_rcnn-based-model.pth) can be downloaded at the release of this project. Then Run the test command:

`python test.py --img  the_tested_img_path/the_image_name.jpg`

# C3D-based-model
## Install 

This code requires you have Keras 2 and TensorFlow 1 or greater installed. Please see the `requirements.txt` file. To ensure you're up to date, run:

`pip install -r requirements.txt`

## Test the trained model
The model(c3d-based-model.hdf5) can be downloaded at the release of this project. Then Run the test command:

`python test.py`
