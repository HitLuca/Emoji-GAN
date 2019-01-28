import sys

from models import utils
from models.began.began_model import BEGAN
from models.cwgan_gp.cwgan_gp_model import CWGAN_GP
from models.dcgan.dcgan_model import DCGAN
from models.run_config import RunConfig
from models.wgan_gp.wgan_gp_model import WGAN_GP
from models.wgan_gp_vae.wgan_gp_vae_model import WGAN_GP_VAE

models_dictionary = {
    'cwgan_gp': CWGAN_GP,
    'wgan_gp': WGAN_GP,
    'dcgan': DCGAN,
    'wgan_gp_vae': WGAN_GP_VAE,
    'began': BEGAN
}


def train(model_type):
    resolution = 16
    channels = 1

    dataset, classes, companies, categories_names = utils.load_emoji_dataset(resolution, channels, shuffle=True)
    classes_n = categories_names.shape[0]
    class_names = categories_names.tolist()

    run_dir, img_dir, model_dir, generated_datasets_dir = utils.generate_run_dir(model_type)

    run_config = RunConfig(classes_n=classes_n,
                           class_names=class_names,
                           channels=channels,
                           resolution=resolution,
                           run_dir=run_dir,
                           img_dir=img_dir,
                           model_dir=model_dir,
                           generated_datasets_dir=generated_datasets_dir)

    model = models_dictionary[model_type](run_config)

    losses = model.train(dataset, classes)

    return losses


def resume_training(model_type, run_folder='../outputs/began/2019-01-19_17-13-37/', checkpoint=3, new_epochs=10):
    resolution, channels = utils.get_dataset_info_from_run(run_folder)

    dataset, classes, companies, categories_names = utils.load_emoji_dataset(resolution, channels, shuffle=True)

    model = models_dictionary[model_type](RunConfig())
    model.resume_training(run_folder, checkpoint, new_epochs)
    losses = model.train(dataset, classes)
    return losses


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'wgan_gp'

    print(model_type)
    train(model_type)
    # resume_training(model_type)
