import json
from typing import Tuple

import numpy as np
from keras.datasets import mnist
from numpy import ndarray
from scipy.misc import imresize


def get_dataset_info_from_run(run_filepath: str) -> Tuple[dict, dict]:
    with open(run_filepath + 'run_config.json', 'r') as f:
        dict = json.load(f)

        return dict['resolution'], dict['channels']


def load_dataset(dataset_folder: str, resolution: int, shuffle: bool = True) -> Tuple[ndarray, ndarray, list, list]:
    dataset: ndarray = np.load(dataset_folder + 'emojis_' + str(resolution) + '.npy')
    classes: ndarray = np.load('dataset/emojis_classes.npy')

    with open(dataset_folder + 'categories_names.json', 'r') as f:
        categories = json.load(f)

    with open(dataset_folder + 'companies_names.json', 'r') as f:
        companies = json.load(f)

    dataset = ((dataset + 1.0) / 2.0)
    alphas = dataset[:, :, :, -1:]
    dataset = dataset[:, :, :, :-1] * alphas + np.ones(dataset.shape)[:, :, :, :-1] * (1 - alphas)

    if shuffle:
        perm = np.random.permutation(dataset.shape[0])
        dataset = dataset[perm]
        classes = classes[perm]

    return dataset, classes, companies, categories


def load_mnist(resolution: int, shuffle: bool = True) -> ndarray:
    (x_train, _), _ = mnist.load_data()
    x_train = np.stack((x_train, x_train, x_train), -1)

    new_images = []
    for i in range(x_train.shape[0]):
        new_images.append(imresize(x_train[i], (resolution, resolution)))

    dataset = np.array(new_images) / 255.0

    if shuffle:
        perm = np.random.permutation(dataset.shape[0])
        dataset = dataset[perm]
    return dataset
