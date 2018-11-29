import numpy as np
import sys

from models import utils
from models.cwgan_gp.cwgan_gp_model import CWGAN_GP
from models.wgan_gp.wgan_gp_model import WGAN_GP
from models.wgan_gp_vae.wgan_gp_vae_model import WGAN_GP_VAE

models_dictionary = {
    'cwgan_gp': CWGAN_GP,
    'wgan_gp': WGAN_GP,
    'wgan_gp_vae': WGAN_GP_VAE
}


def train(model_type):
    batch_size = 32
    resolution = 16
    channels = 4

    companies = np.load('../dataset/companies_names.npy')
    categories_names = np.load('../dataset/categories_names.npy')

    dataset = np.load('../dataset/emojis_' + str(resolution) + '.npy')
    classes = np.load('../dataset/emojis_classes.npy')[:, 1]
    classes_n = categories_names.shape[0]

    perm = np.random.permutation(dataset.shape[0])
    dataset = dataset[perm]
    classes = classes[perm]

    run_dir, img_dir, model_dir, generated_datesets_dir = utils.generate_run_dir(model_type)

    config_2 = {
        'batch_size': batch_size,
        'classes_n': classes_n,
        'class_names': categories_names.tolist(),
        'channels': channels,
        'resolution': resolution,
        'run_dir': run_dir,
        'img_dir': img_dir,
        'model_dir': model_dir,
        'generated_datesets_dir': generated_datesets_dir,
    }

    config = utils.merge_config_and_save(config_2)

    model = models_dictionary[model_type](config)
    losses = model.train(dataset, classes)
    return losses


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'vae'
    print(model_type)
    train(model_type)
