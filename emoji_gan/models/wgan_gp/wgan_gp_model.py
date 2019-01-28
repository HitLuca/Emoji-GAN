import os
import pickle

from keras.models import load_model

from models.wgan_gp import wgan_gp_utils
from models import utils
import numpy as np


class WGAN_GP:
    def __init__(self, config, restore_models=False, models_root_filepath='', new_epochs=0):
        if restore_models:
            self._restore_models(models_root_filepath, new_epochs)
        else:
            self._apply_config(config)
            self._losses = [[], []]
            self._build_models()

    def _apply_config(self, config):
        self._channels = config['channels']
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']
        self._resolution = config['resolution']
        self._n_critic = config['n_critic']

        self._n_generator = config['n_generator']
        self._latent_dim = config['latent_dim']

        self._generator_lr = config['generator_lr']
        self._critic_lr = config['critic_lr']
        self._img_frequency = config['img_frequency']
        self._loss_frequency = config['loss_frequency']
        self._latent_space_frequency = config['latent_space_frequency']
        self._model_save_frequency = config['model_save_frequency']
        self._dataset_generation_frequency = config['dataset_generation_frequency']
        self._dataset_generation_size = config['dataset_generation_size']
        self._gradient_penalty_weight = config['gradient_penalty_weight']
        self._run_dir = config['run_dir']
        self._img_dir = config['img_dir']
        self._model_dir = config['model_dir']
        self._generated_datasets_dir = config['generated_datasets_dir']

        self._lr_decay_factor = config['lr_decay_factor']
        self._lr_decay_steps = config['lr_decay_steps']
        self._epoch = config['epoch']

    def _restore_models(self, models_root_filepath, new_epochs):
        config_filepath = models_root_filepath + 'config.json'
        config = utils.load_config(config_filepath)

        self._apply_config(config)
        self._epochs = new_epochs

        losses_filepath = models_root_filepath + 'losses.p'
        with open(losses_filepath, 'rb') as f:
            self._losses = pickle.load(f)
            self._losses[0] = self._losses[0][:self._epoch]
            self._losses[1] = self._losses[1][:self._epoch]

        self._build_models()

        model_checkpoints = [f.path for f in os.scandir(models_root_filepath + 'models/') if f.is_dir()]
        model_checkpoint = sorted(model_checkpoints)[-1]
        self._load_models(model_checkpoint + '/')

    def _build_models(self):
        self._generator = wgan_gp_utils.build_generator(self._latent_dim, self._resolution, self._channels)
        self._critic = wgan_gp_utils.build_critic(self._resolution, self._channels)
        self._generator_model = wgan_gp_utils.build_generator_model(self._generator, self._critic, self._latent_dim,
                                                                    self._generator_lr)
        self._critic_model = wgan_gp_utils.build_critic_model(self._generator, self._critic, self._latent_dim,
                                                              self._resolution, self._channels, self._batch_size,
                                                              self._critic_lr, self._gradient_penalty_weight)

    def train(self, dataset, *_):
        ones = np.ones((self._batch_size, 1))
        neg_ones = -ones
        zeros = np.zeros((self._batch_size, 1))

        while self._epoch < self._epochs:
            self._epoch += 1
            critic_losses = []
            for _ in range(self._n_critic):
                indexes = np.random.randint(0, dataset.shape[0], self._batch_size)
                real_samples = dataset[indexes]
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [real_samples, noise]

                critic_losses.append(self._critic_model.train_on_batch(inputs, [ones, neg_ones, zeros])[0])
            critic_loss = np.mean(critic_losses)

            generator_losses = []
            for _ in range(self._n_generator):
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [noise]

                generator_losses.append(self._generator_model.train_on_batch(inputs, ones))
            generator_loss = np.mean(generator_losses)

            generator_loss = float(-generator_loss)
            critic_loss = float(-critic_loss)

            self._losses[0].append(generator_loss)
            self._losses[1].append(critic_loss)

            # if self._epoch % 100 == 0:
            print("%d [C loss: %+.6f] [G loss: %+.6f]" % (self._epoch, critic_loss, generator_loss))

            if self._epoch % self._loss_frequency == 0:
                self._save_losses()

            if self._epoch % self._img_frequency == 0:
                self._save_samples()

            if self._epoch % self._latent_space_frequency == 0:
                self._save_latent_space()

            if self._epoch % self._model_save_frequency == 0:
                self._save_models()

            if self._epoch % self._dataset_generation_frequency == 0:
                self._generate_dataset()

            if self._epoch % self._lr_decay_steps == 0:
                self._apply_lr_decay()

        self._generate_dataset()
        self._save_losses()
        self._save_models()
        self._save_samples()
        self._save_latent_space()

        return self._losses

    def _save_samples(self):
        rows, columns = 6, 6
        noise = np.random.normal(0, 1, (rows * columns, self._latent_dim))
        generated_samples = self._generator.predict(noise)

        filenames = [self._img_dir + ('/%07d.png' % self._epoch), self._img_dir + '/last.png']
        utils.save_samples(generated_samples, rows, columns, self._resolution, self._channels, filenames)

    def _save_latent_space(self):
        grid_size = 6

        latent_space_inputs = np.zeros((grid_size * grid_size, self._latent_dim))

        for i, v_i in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
            for j, v_j in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
                latent_space_inputs[i * grid_size + j, :2] = [v_i, v_j]

        generated_samples = self._generator.predict(latent_space_inputs)

        filenames = [self._img_dir + '/latent_space.png', self._img_dir + ('/%07d_latent_space.png' % self._epoch)]
        utils.save_latent_space(generated_samples, grid_size, self._resolution, self._channels, filenames)

    def _save_losses(self):
        utils.save_losses_wgan(self._losses, self._img_dir + '/losses.png')

        with open(self._run_dir + '/losses.p', 'wb') as f:
            pickle.dump(self._losses, f)

    def _save_models(self):
        root_dir = self._model_dir + '/' + str(self._epoch) + '/'
        os.makedirs(root_dir)
        self._generator_model.save_weights(root_dir + 'generator_model.h5')
        self._critic_model.save_weights(root_dir + 'critic_model.h5')
        self._generator.save_weights(root_dir + 'generator.h5')
        self._critic.save_weights(root_dir + 'critic.h5')

        utils.update_config_epoch(self._run_dir + '/config.json', self._epoch)

    def _load_models(self, models_checkpoint):
        self._generator_model.load_weights(models_checkpoint + 'generator_model.h5')
        self._critic_model.load_weights(models_checkpoint + 'critic_model.h5')
        self._generator.load_weights(models_checkpoint + 'generator.h5')
        self._critic.load_weights(models_checkpoint + 'critic.h5')

    def _generate_dataset(self):
        z_samples = np.random.normal(0, 1, (self._dataset_generation_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datasets_dir + ('/%d_generated_data' % self._epoch), generated_dataset)

    def get_models(self):
        return self._generator, self._critic, self._generator_model, self._critic_model

    def _apply_lr_decay(self):
        models = [self._generator_model, self._critic_model]
        utils.apply_lr_decay(models, self._lr_decay_factor)
