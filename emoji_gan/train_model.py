import sys

from emoji_gan import data_utils
from emoji_gan.data_utils import generate_run_dir
from gan_collection.models.began.began_model import BEGAN
from gan_collection.models.cwgan_gp.cwgan_gp_model import CWGAN_GP
from gan_collection.models.dcgan.dcgan_model import DCGAN
from gan_collection.models.wgan_gp.wgan_gp_model import WGAN_GP
from gan_collection.models.wgan_gp_vae.wgan_gp_vae_model import WGAN_GP_VAE

RESOLUTION = 16
CHANNELS = 3
DATASET_FOLDER = '../dataset/'
OUTPUTS_FOLDER = '../outputs/'

EPOCHS = 100000
LOSS_SAVE_FREQUENCY = 125
OUTPUT_SAVE_FREQUENCY = 500
LATENT_SPACE_SAVE_FREQUENCY = 1000
MODEL_SAVE_FREQUENCY = -1
DATASET_GENERATION_FREQUENCY = -1
DATASET_SIZE = 1000

LATENT_DIM = 10


def train(model_type: str) -> None:
    dataset, classes, companies, categories = data_utils.load_emoji_dataset(DATASET_FOLDER, RESOLUTION, shuffle=True)

    dataset_companies, dataset_categories = classes[:, 0], classes[:, 1]
    companies_n, categories_n = len(companies), len(categories)

    # dataset = data_utils.load_mnist(RESOLUTION)
    # CHANNELS = 1

    run_dir, outputs_dir, model_dir, generated_datasets_dir = generate_run_dir(OUTPUTS_FOLDER, model_type)

    if model_type == 'wgan_gp':
        model = WGAN_GP(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                        generated_datasets_dir=generated_datasets_dir, resolution=RESOLUTION,
                        channels=CHANNELS, epochs=EPOCHS, output_save_frequency=OUTPUT_SAVE_FREQUENCY,
                        model_save_frequency=MODEL_SAVE_FREQUENCY, loss_save_frequency=LOSS_SAVE_FREQUENCY,
                        latent_space_save_frequency=LATENT_SPACE_SAVE_FREQUENCY,
                        dataset_generation_frequency=DATASET_GENERATION_FREQUENCY,
                        dataset_size=DATASET_SIZE, latent_dim=LATENT_DIM)
    elif model_type == 'dcgan':
        model = DCGAN(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                      generated_datasets_dir=generated_datasets_dir, resolution=RESOLUTION,
                      channels=CHANNELS, epochs=EPOCHS, output_save_frequency=OUTPUT_SAVE_FREQUENCY,
                      model_save_frequency=MODEL_SAVE_FREQUENCY, loss_save_frequency=LOSS_SAVE_FREQUENCY,
                      latent_space_save_frequency=LATENT_SPACE_SAVE_FREQUENCY,
                      dataset_generation_frequency=DATASET_GENERATION_FREQUENCY,
                      dataset_size=DATASET_SIZE, latent_dim=LATENT_DIM)
    elif model_type == 'cwgan_gp':
        model = CWGAN_GP(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                         generated_datasets_dir=generated_datasets_dir, resolution=RESOLUTION,
                         channels=CHANNELS, epochs=EPOCHS, output_save_frequency=OUTPUT_SAVE_FREQUENCY,
                         model_save_frequency=MODEL_SAVE_FREQUENCY, loss_save_frequency=LOSS_SAVE_FREQUENCY,
                         latent_space_save_frequency=LATENT_SPACE_SAVE_FREQUENCY,
                         dataset_generation_frequency=DATASET_GENERATION_FREQUENCY,
                         dataset_size=DATASET_SIZE, classes=categories,
                         classes_n=categories_n, latent_dim=LATENT_DIM)
    elif model_type == 'wgan_gp_vae':
        model = WGAN_GP_VAE(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                            generated_datasets_dir=generated_datasets_dir, resolution=RESOLUTION,
                            channels=CHANNELS, epochs=EPOCHS, output_save_frequency=OUTPUT_SAVE_FREQUENCY,
                            model_save_frequency=MODEL_SAVE_FREQUENCY, loss_save_frequency=LOSS_SAVE_FREQUENCY,
                            latent_space_save_frequency=LATENT_SPACE_SAVE_FREQUENCY,
                            dataset_generation_frequency=DATASET_GENERATION_FREQUENCY,
                            dataset_size=DATASET_SIZE, latent_dim=LATENT_DIM)
    elif model_type == 'began':
        model = BEGAN(run_dir=run_dir, outputs_dir=outputs_dir, model_dir=model_dir,
                      generated_datasets_dir=generated_datasets_dir, resolution=RESOLUTION,
                      channels=CHANNELS, epochs=EPOCHS, output_save_frequency=OUTPUT_SAVE_FREQUENCY,
                      model_save_frequency=MODEL_SAVE_FREQUENCY, loss_save_frequency=LOSS_SAVE_FREQUENCY,
                      latent_space_save_frequency=LATENT_SPACE_SAVE_FREQUENCY,
                      dataset_generation_frequency=DATASET_GENERATION_FREQUENCY,
                      dataset_size=DATASET_SIZE, latent_dim=LATENT_DIM)
    else:
        raise NotImplementedError

    model.train(dataset, dataset_categories)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'wgan_gp'

    print(model_type)
    train(model_type)
