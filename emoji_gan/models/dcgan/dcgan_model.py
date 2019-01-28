import os
import pickle

import numpy as np

from models import utils
from models.dcgan import dcgan_utils


class DCGAN:
    def __init__(self, config):
        self._channels = config['channels']
        self._batch_size = config['batch_size']
        self._epochs = config['epochs']
        self._resolution = config['resolution']
        self._n_discriminator = config['n_critic']

        self._n_generator = config['n_generator']
        self._latent_dim = config['latent_dim']

        self._generator_lr = config['generator_lr']
        self._discriminator_lr = config['critic_lr']
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

        self._epoch = 0
        self._losses = [[], []]
        self._build_models()

    def _build_models(self):
        self._generator = dcgan_utils.build_generator(self._latent_dim, self._resolution, self._channels)
        self._discriminator = dcgan_utils.build_discriminator(self._resolution, self._channels)
        self._generator_model = dcgan_utils.build_generator_model(self._generator, self._discriminator,
                                                                  self._latent_dim,
                                                                  self._generator_lr)
        self._discriminator_model = dcgan_utils.build_discriminator_model(self._generator, self._discriminator,
                                                                          self._latent_dim,
                                                                          self._resolution, self._channels,
                                                                          self._discriminator_lr)

    def train(self, dataset, *_):
        ones = np.ones((self._batch_size, 1))
        neg_ones = -ones
        zeros = np.zeros((self._batch_size, 1))

        while self._epoch < self._epochs:
            self._epoch += 1
            discriminator_losses = []
            for _ in range(self._n_discriminator):
                indexes = np.random.randint(0, dataset.shape[0], self._batch_size)
                real_samples = dataset[indexes]
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [real_samples, noise]

                discriminator_losses.append(
                    self._discriminator_model.train_on_batch(inputs, [ones, zeros])[0])
            discriminator_loss = np.mean(discriminator_losses)

            generator_losses = []
            for _ in range(self._n_generator):
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [noise]

                generator_losses.append(self._generator_model.train_on_batch(inputs, ones))
            generator_loss = np.mean(generator_losses)

            generator_loss = float(generator_loss)
            discriminator_loss = float(discriminator_loss)

            self._losses[0].append(generator_loss)
            self._losses[1].append(discriminator_loss)

            if self._epoch % 100 == 0:
                print("%d [D loss: %+.6f] [G loss: %+.6f]" % (self._epoch, discriminator_loss, generator_loss))

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
        self._generator_model.save(root_dir + 'generator_model.h5')
        self._discriminator_model.save(root_dir + 'discriminator_model.h5')
        self._generator.save(root_dir + 'generator.h5')
        self._discriminator.save(root_dir + 'discriminator.h5')

        utils.update_config_epoch(self._run_dir + '/config.json', self._epoch)

    def _generate_dataset(self):
        z_samples = np.random.normal(0, 1, (self._dataset_generation_size, self._latent_dim))
        generated_dataset = self._generator.predict(z_samples)
        np.save(self._generated_datasets_dir + ('/%d_generated_data' % self._epoch), generated_dataset)

    def get_models(self):
        return self._generator, self._discriminator, self._generator_model, self._discriminator_model

    def _apply_lr_decay(self):
        models = [self._generator_model, self._discriminator_model]
        utils.apply_lr_decay(models, self._lr_decay_factor)
