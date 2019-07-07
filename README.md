# Emoji-GAN
[![HitCount](http://hits.dwyl.io/HitLuca/Emoji-GAN.svg)](http://hits.dwyl.io/HitLuca/Emoji-GAN)
[![Total alerts](https://img.shields.io/lgtm/alerts/g/HitLuca/Emoji-GAN.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HitLuca/Emoji-GAN/alerts/)
[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/HitLuca/Emoji-GAN.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/HitLuca/Emoji-GAN/context:python)

## Description
Using generative adversarial networks to generate emojis

### Dataset
For this project I used the scraped emojis available in the [emoji-data](https://github.com/iamcal/emoji-data) repo.

The data used comes in 4 different resolutions, 16, 20, 32 and 64 pixels RGBA.

## Project structure
The [```dataset```](dataset) folder contains the various datasets to be used during training.

The [```emoij_gan```](emoji_gan) folder contains all the code necessary to download the dataset and train the models

Outputs of the various runs will be stored in the [```outputs```]() folder.

## Getting started
This project is intended to be self-contained, with the only exception being the dataset that is downloaded automatically.

Before starting, run the [```setup_dataset.py```](emoji_gan/setup_dataset.py) script, that will automatically download and parse the dataset, creating ready-to-use .npy files.

### Prerequisites
To install the python environment for this project, refer to the [Pipenv setup guide](https://pipenv.readthedocs.io/en/latest/basics/)

### Training the models
Just choose your model of choice and give it as an argument to the the [```train_model.py```](emoji_gan/train_model.py) script. 
