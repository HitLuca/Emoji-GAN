# Emoji-GAN

## Description
Generative Adversarial Network models tasked with generating realistic emojis sourced from popular companies (Google, Facebook etc.).

### Dataset
For this project I used the scraped emojis available in the [emoji-data](https://github.com/iamcal/emoji-data) repo.

The data used comes in 4 different resolutions, 16, 20, 32 and 64 pixels RGBA.

### Models
At this point three different generative models are implemented:

* Conditional Improved Wasserstein GAN (CWGAN-GP)
* Improved Wasserstein GAN (WGAN-GP)
* Variational Autoencoder with learned similarity metric (VAE with l.s.m., WGAN-GP-VAE)

The WGAN-GP-VAE model implementation comes from my [other](https://github.com/HitLuca/GANs_for_spiking_time_series) project, as a part of my Master Thesis (AI)

## Project structure
The [```models```](emoji_gan/models) folder contains all the generative models implemented.

The [```dataset```](dataset) folder contains the various datasets to be used during training.

## Getting started
This project is intended to be self-contained, with the only exception being the dataset that is downloaded automatically.

Before starting, run the [```setup.py```](setup.py) script, that will automatically download and parse the dataset, creating ready-to-use .npy files.

### Prerequisites
To install the python environment for this project, refer to the [Pipenv setup guide](https://pipenv.readthedocs.io/en/latest/basics/)

### Training the models
Just choose your model of choice and give it as an argument to the the [```train_model.py```](emoji_gan/train_model.py) script. 
Model parameters that can be changed are located in [```train_model.py```](emoji_gan/train_model.py) and [```utils.py```](emoji_gan/models/utils.py) scripts, but can also be overwritten by adding the new value to the configuration dictionary in the training file.
