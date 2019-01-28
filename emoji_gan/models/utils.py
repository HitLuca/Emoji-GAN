import json
import os
from datetime import datetime

import keras.backend as K
import numpy as np
# uncomment to use on machines that don't support standard matplotlib backend
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt


def set_model_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


def save_samples(generated_data, rows, columns, resolution, channels, filenames, labels=None):
    plt.subplots(rows, columns, figsize=(7, 7))

    k = 1
    for i in range(rows):
        for j in range(columns):
            plt.subplot(rows, columns, k)
            if labels:
                plt.title(labels[k - 1])
            if channels == 1:
                plt.imshow((generated_data[k - 1].reshape(resolution, resolution) + 1.0) / 2.0, cmap='gray')
            else:
                plt.imshow((generated_data[k - 1].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            plt.xticks([])
            plt.yticks([])
            k += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def save_losses(losses, filename, legend_names):
    plt.figure(figsize=(15, 4.5))
    for loss, legend_name in zip(losses, legend_names):
        plt.plot(loss, label=legend_name)
    plt.legend()
    plt.savefig(filename)
    plt.clf()
    plt.close()


def save_latent_space(generated_data, grid_size, resolution, channels, filenames, title=None):
    plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    if title:
        plt.suptitle(title)

    for i in range(grid_size):
        for j in range(grid_size):
            plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
            if channels == 1:
                plt.imshow((generated_data[i * grid_size + j].reshape(resolution, resolution) + 1.0) / 2.0, cmap='gray')
            else:
                plt.imshow((generated_data[i * grid_size + j].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def generate_run_dir(model_type):
    root_path = '../outputs/' + model_type
    if not os.path.exists(root_path):
        os.makedirs(root_path)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_dir = root_path + '/' + current_datetime
    img_dir = run_dir + '/img'
    model_dir = run_dir + '/models'
    generated_datasets_dir = run_dir + '/generated_datasets'

    os.mkdir(run_dir)
    os.mkdir(img_dir)
    os.mkdir(model_dir)
    os.mkdir(generated_datasets_dir)

    return run_dir, img_dir, model_dir, generated_datasets_dir


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def apply_lr_decay(models, lr_decay_factor):
    for model in models:
        lr_tensor = model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * lr_decay_factor)


def load_emoji_dataset(resolution, channels, shuffle):
    companies = np.load('../dataset/companies_names.npy')
    categories_names = np.load('../dataset/categories_names.npy')

    dataset = np.load('../dataset/emojis_' + str(resolution) + '.npy')
    classes = np.load('../dataset/emojis_classes.npy')[:, 1]

    if shuffle:
        perm = np.random.permutation(dataset.shape[0])
        dataset = dataset[perm]
        classes = classes[perm]

    if channels == 1 or channels == 3:
        dataset = (dataset + 1) / 2.0
        alphas = dataset[:, :, :, -1:]

        if channels == 1:
            dataset = np.expand_dims(np.dot(dataset[:, :, :, :3], [0.299, 0.587, 0.114]), -1)
            dataset = dataset * alphas + np.ones(dataset.shape) * (1-alphas)
        elif channels == 3:
            dataset = dataset[:, :, :, :-1] * alphas + np.ones(dataset.shape)[:, :, :, :-1] * (1-alphas)
        dataset = (dataset * 2) - 1.0

    return dataset, classes, companies, categories_names


def get_dataset_info_from_run(run_filepath):
    with open(run_filepath + 'run_config.json', 'r') as f:
        dict = json.load(f)

        return dict['resolution'], dict['channels']
