import json

from models.abstract_gan.abstract_gan_config import AbstractGANConfig


class RunConfig(AbstractGANConfig):
    def __init__(self, classes_n=0,
                 class_names=None,
                 channels=3,
                 resolution=16,
                 run_dir='',
                 img_dir='',
                 model_dir='',
                 generated_datasets_dir=''):
        super().__init__()
        self.epochs = 100
        self.img_frequency = 250
        self.loss_frequency = 125
        self.latent_space_frequency = 1000
        self.model_save_frequency = 1000
        self.dataset_generation_frequency = 1000
        self.dataset_generation_size = 1000
        self.epoch = 0

        self.classes_n = classes_n
        self.class_names = class_names
        self.channels = channels
        self.resolution = resolution
        self.run_dir = run_dir
        self.img_dir = img_dir
        self.model_dir = model_dir
        self.generated_datasets_dir = generated_datasets_dir

    def restore(self, config_filepath):
        with open(config_filepath, 'r') as f:
            dict = json.load(f)

        self.epochs = dict['epochs']
        self.img_frequency = dict['img_frequency']
        self.loss_frequency = dict['loss_frequency']
        self.latent_space_frequency = dict['latent_space_frequency']
        self.model_save_frequency = dict['model_save_frequency']
        self.dataset_generation_frequency = dict['dataset_generation_frequency']
        self.dataset_generation_size = dict['dataset_generation_size']
        self.epoch = dict['epoch']

        self.classes_n = dict['classes_n']
        self.class_names = dict['class_names']
        self.channels = dict['channels']
        self.resolution = dict['resolution']
        self.run_dir = dict['run_dir']
        self.img_dir = dict['img_dir']
        self.model_dir = dict['model_dir']
        self.generated_datasets_dir = dict['generated_datasets_dir']
