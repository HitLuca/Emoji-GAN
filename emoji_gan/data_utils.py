import json
import os
from datetime import datetime
from typing import Tuple

import numpy as np
from keras.datasets import mnist
from numpy import ndarray
from scipy.misc import imresize


def get_dataset_info_from_run(run_filepath: str) -> Tuple[dict, dict]:
    with open(run_filepath + 'run_config.json', 'r') as f:
        dict = json.load(f)

        return dict['resolution'], dict['channels']


def load_emoji_dataset(dataset_folder: str, resolution: int, shuffle: bool = True) -> Tuple[ndarray, ndarray, list, list]:
    dataset: ndarray = np.load(dataset_folder + 'emojis_' + str(resolution) + '.npy')
    classes: ndarray = np.load(dataset_folder + 'emojis_classes.npy')

    with open(dataset_folder + 'categories_names.json', 'r') as f:
        categories = json.load(f)

    with open(dataset_folder + 'companies_names.json', 'r') as f:
        companies = json.load(f)

    alphas = dataset[:, :, :, -1:]
    dataset = dataset[:, :, :, :-1] * alphas + np.ones(dataset.shape)[:, :, :, :-1] * (1 - alphas)

    if shuffle:
        perm = np.random.permutation(dataset.shape[0])
        dataset = dataset[perm]
        classes = classes[perm]

    return dataset, classes, companies, categories


def load_mnist(resolution: int, shuffle: bool = True) -> ndarray:
    (x_train, _), _ = mnist.load_data()

    new_images = []
    for i in range(x_train.shape[0]):
        new_images.append(imresize(x_train[i], (resolution, resolution)))

    dataset = np.expand_dims(np.array(new_images) / 255.0, -1)

    if shuffle:
        perm = np.random.permutation(dataset.shape[0])
        dataset = dataset[perm]
    return dataset


def generate_run_dir(path: str, model_type):
    root_path = path + model_type + '/'
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_dir = root_path + current_datetime + '/'
    outputs_dir = run_dir + 'outputs/'
    model_dir = run_dir + 'models/'
    generated_datasets_dir = run_dir + 'generated_datasets/'

    os.mkdir(run_dir)
    os.mkdir(outputs_dir)
    os.mkdir(model_dir)
    os.mkdir(generated_datasets_dir)

    return run_dir, outputs_dir, model_dir, generated_datasets_dir
