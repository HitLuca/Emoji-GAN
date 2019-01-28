import json

from models.abstract_gan.abstract_gan_config import AbstractGANConfig


class BEGANConfig(AbstractGANConfig):
    def __init__(self):
        super().__init__()
        self.gamma = 0.5
        self.latent_dim = 64
        self.lambda_k = 0.001
        self.batch_size = 16
        self.initial_lr = 0.0001
        self.min_lr = 0.00001
        self.lr = self.initial_lr
        self.lr_decay_rate = 0.9
        self.k = 0
        self.loss_exponent = 1

    def restore(self, config_filepath):
        with open(config_filepath) as f:
            dict = json.load(f)

        self.gamma = dict['gamma']
        self.latent_dim = dict['latent_dim']
        self.lambda_k = dict['lambda_k']
        self.batch_size = dict['batch_size']
        self.initial_lr = dict['initial_lr']
        self.min_lr = dict['min_lr']
        self.lr = dict['lr']
        self.lr_decay_rate = dict['lr_decay_rate']
        self.k = dict['k']
        self.loss_exponent = dict['loss_exponent']
