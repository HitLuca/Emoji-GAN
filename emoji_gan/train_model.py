import numpy as np
import sys

from models import utils
from models.cwgan_gp.cwgan_gp_model import CWGAN_GP
from models.wgan_gp.wgan_gp_model import WGAN_GP
from models.ds_wgan_gp.ds_wgan_gp_model import DS_WGAN_GP
from models.wgan_gp_vae.wgan_gp_vae_model import WGAN_GP_VAE

models_dictionary = {
    'cwgan_gp': CWGAN_GP,
    'wgan_gp': WGAN_GP,
    'ds_wgan_gp': DS_WGAN_GP,
    'wgan_gp_vae': WGAN_GP_VAE
}


def train(model_type):
    resolution = 16
    channels = 3

    companies = np.load('../dataset/companies_names.npy')
    categories_names = np.load('../dataset/categories_names.npy')

    dataset = np.load('../dataset/emojis_' + str(resolution) + '.npy')
    classes = np.load('../dataset/emojis_classes.npy')[:, 1]
    classes_n = categories_names.shape[0]

    if channels == 1 or channels == 3:
        dataset = (dataset + 1) / 2.0
        alphas = dataset[:, :, :, -1:]

        if channels == 1:
            dataset = np.expand_dims(np.dot(dataset[:, :, :, :3], [0.299, 0.587, 0.114]), -1)
            dataset = dataset * alphas + np.ones(dataset.shape) * (1-alphas)
        elif channels == 3:
            dataset = dataset[:, :, :, :-1] * alphas + np.ones(dataset.shape)[:, :, :, :-1] * (1-alphas)
        dataset = (dataset * 2) - 1.0

    perm = np.random.permutation(dataset.shape[0])
    dataset = dataset[perm]
    classes = classes[perm]

    run_dir, img_dir, model_dir, generated_datasets_dir = utils.generate_run_dir(model_type)

    config_2 = {
        'classes_n': classes_n,
        'class_names': categories_names.tolist(),
        'channels': channels,
        'resolution': resolution,
        'run_dir': run_dir,
        'img_dir': img_dir,
        'model_dir': model_dir,
        'generated_datasets_dir': generated_datasets_dir,
    }

    config = utils.merge_config_and_save(config_2)

    model = models_dictionary[model_type](config)
    losses = model.train(dataset, classes)
    return losses


def train_ds_wgan_gp():
    resolution = 16

    companies = np.load('../dataset/companies_names.npy')
    categories_names = np.load('../dataset/categories_names.npy')

    dataset = np.load('../dataset/emojis_' + str(resolution) + '.npy')
    classes = np.load('../dataset/emojis_classes.npy')[:, 1]
    classes_n = categories_names.shape[0]

    dataset = (dataset + 1) / 2.0
    alphas = dataset[:, :, :, -1:]

    dataset_grayscale = np.expand_dims(np.dot(dataset[:, :, :, :3], [0.299, 0.587, 0.114]), -1)
    dataset_grayscale = dataset_grayscale * alphas + np.ones(dataset_grayscale.shape) * (1 - alphas)
    dataset_grayscale = (dataset_grayscale * 2) - 1.0

    dataset_rgb = dataset[:, :, :, :-1] * alphas + np.ones(dataset.shape)[:, :, :, :-1] * (1-alphas)
    dataset_rgb = (dataset_rgb * 2) - 1.0

    perm = np.random.permutation(dataset.shape[0])
    dataset_grayscale = dataset_grayscale[perm]
    dataset_rgb = dataset_rgb[perm]
    classes = classes[perm]

    run_dir, img_dir, model_dir, generated_datasets_dir = utils.generate_run_dir('ds_wgan_gp')

    config_2 = {
        'classes_n': classes_n,
        'class_names': categories_names.tolist(),
        'resolution': resolution,
        'run_dir': run_dir,
        'img_dir': img_dir,
        'model_dir': model_dir,
        'generated_datasets_dir': generated_datasets_dir,
    }

    config = utils.merge_config_and_save(config_2)

    model = models_dictionary['ds_wgan_gp'](config)
    losses = model.train(dataset_grayscale, dataset_rgb)
    return losses


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'cwgan_gp'
    train(model_type)
    # train_ds_wgan_gp()
