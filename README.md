# Grayscale Image Colorization

A deep-learning project for colorization of grayscale images.

## Overview

The project uses a U-Net-style convolutional neural network implemented
with PyTorch.

Images are converted to CIELAB color space. The L channel represents
the grayscale input, while the neural network predicts the a and b
chrominance channels.

The final color image is reconstructed by combining the original
L channel with the predicted a and b channels and converting the
result from LAB to RGB.

## Model

- Architecture: U-Net
- Input: Grayscale L channel
- Output: a and b color channels
- Image size during training: 96x96
- Framework: PyTorch
- Training: From scratch
- Final checkpoint: stage4_landscape.pth
- Dataset: Landscape images

## Main Files

- `model.py` - neural network architecture
- `dataset.py` - dataset loading and preprocessing
- `losses.py` - colorization loss
- `train.py` - model training
- `evaluate.py` - model evaluation
- `interference.py` - colorization of new grayscale images
- `prepare_landscape_data.py` - prepares the landscape dataset

## Usage

Place a grayscale image in the project and specify its path in
`inference.py`.

Run:

python interference.py

The colorized result will be saved in the outputs folder.