# Road Segmentation &mdash; Computational Intelligence Laboratory Project 2018
This repository summarises our findings and models we used for the task of road segmentation from RGB aerial/satellite images.

All questions should be directed towards the authors in case discrepancies are found or our README is incomplete in
instructions.

## Setup
The following steps need to be followed to train our networks.

### Download Necessary Dataset
Download the training and testing data from the Kaggle page and unzip them into `data/`.

### Installing Dependencies
The following dependencies needed to be fulfilled:
  * A fixed [Keras-2.2.0](https://github.com/mvaenskae/keras)
  * Tensorflow-1.7.0 or newer
  * [imgaug-0.2.5](https://github.com/aleju/imgaug)

We make brief notes of installation instructions and make references to official installation instructions when possible.

#### Installing Keras-2.2.0
Checkout branch `cil-road-segmentation-2018` and follow normal installation instructions provided by official Keras.

#### Installing Tensorflow-1.7.0
Install Tensorflow with GPU support. This is required as we make heavy use of NCHW and have not verified NHWC to work.

#### Installing imgaug-0.2.5
Follow the installation instructions given by imgaug.

#### Environment Setup
We further require the user to create a configuration file for keras at `~/.keras/keras.json` with the following content:
```
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "tensorflow",
    "image_data_format": "channels_first"
}
```

## Running Networks
We have the following selection of networks for quick training:
  * ResNet18 (`resnet18`)
  * SegNet (`segnet`)
  * RedNet50 (`rednet50`)

All networks are called with their own `train-*.py` file for training and `predict-*.py` file for generating predictions. Their source code can be found in the respective imports.

ResNet18 generates a ready-to-use `.csv`-file. SegNet and RedNet50 require the use of `data/mask_to_submission.py` to convert predicted masks to a `.csv`-file.

## Report
The report including source files is in `./report`. We publish this work using the Attribution-NonCommercial-ShareAlike
(CC BY-NC-SA) license.

## Code License
See `LICENSE`.  
