import os
import pickle

import keras.backend as K
import numpy as np

from models.abstract_gan.abstract_gan_model import AbstractGAN
from models.began import began_utils
from models.began.began_config import BEGANConfig
from models.run_config import RunConfig


class BEGAN(AbstractGAN):
    def __init__(self, r_c, m_c=BEGANConfig()):
        super().__init__(r_c, m_c)
        self._losses = [[], [], [], []]
        self._legend_names = ['generator', 'discriminator', 'k', 'm_global']

        self._build_models()

    def train(self, dataset, *_):
        zeros = np.zeros(self._m_c.batch_size)
        zeros_2 = np.zeros(self._m_c.batch_size * 2)

        batches_per_epoch = dataset.shape[0] // self._m_c.batch_size
        dataset_indexes = np.arange(len(dataset))

        last_m_global = np.inf
        lr_decay_step = 0

        while self._r_c.epoch < self._r_c.epochs:
            self._r_c.epoch += 1
            np.random.shuffle(dataset_indexes)

            self._m_c.lr = max(self._m_c.initial_lr * (self._m_c.lr_decay_rate ** lr_decay_step), self._m_c.min_lr)
            K.set_value(self._generator_model.optimizer.lr, self._m_c.lr)
            K.set_value(self._discriminator_model.optimizer.lr, self._m_c.lr)

            m_history = []
            k_history = []
            generator_losses = []
            discriminator_losses = []
            for batch in range(batches_per_epoch):
                k_model = np.ones(self._m_c.batch_size) * self._m_c.k

                indexes = dataset_indexes[batch * self._m_c.batch_size:(batch + 1) * self._m_c.batch_size]
                real_samples = dataset[indexes]

                noise_decoder = np.random.uniform(-1, 1, (self._m_c.batch_size, self._m_c.latent_dim))
                noise_generator = np.random.uniform(-1, 1, (self._m_c.batch_size * 2, self._m_c.latent_dim))

                generated_samples = self._generator.predict(noise_decoder)

                discriminator_loss_real, discriminator_loss_generated = self._discriminator_model.predict(
                    [real_samples, generated_samples, k_model])
                self._discriminator_model.train_on_batch([real_samples, generated_samples, k_model], [zeros, zeros])

                generator_loss = self._generator_model.train_on_batch(noise_generator, zeros_2)

                discriminator_loss_real = np.mean(discriminator_loss_real)
                discriminator_loss_generated = np.mean(discriminator_loss_generated)

                discriminator_loss = float(discriminator_loss_real - self._m_c.k * discriminator_loss_generated)

                self._update_k(discriminator_loss_real, generator_loss)

                m_value = discriminator_loss_real + np.abs(self._m_c.gamma * discriminator_loss_real - generator_loss)
                m_history.append(m_value)
                generator_losses.append(generator_loss)
                discriminator_losses.append(discriminator_loss)
                k_history.append(self._m_c.k)

                self._losses[0].append(generator_loss)
                self._losses[1].append(discriminator_loss)
                self._losses[2].append(self._m_c.k)
                self._losses[3].append(m_value)

            generator_loss = float(np.mean(generator_losses))
            discriminator_loss = float(np.mean(discriminator_losses))
            k = float(np.mean(k_history))
            m_global = float(np.mean(m_history))

            if last_m_global <= m_global:  # decay LearningRate
                lr_decay_step += 1
            last_m_global = m_global

            print("%d [D loss: %+.6f] [G loss: %+.6f] [K: %+.6f] [M: %+.6f]" %
                  (self._r_c.epoch, discriminator_loss, generator_loss, k, m_global))
            print("[learning_rate: %+.6f]" % self._m_c.lr)

            if self._r_c.epoch % self._r_c.loss_frequency == 0:
                self._save_losses()

            if self._r_c.epoch % self._r_c.img_frequency == 0:
                self._save_samples()

            if self._r_c.epoch % self._r_c.latent_space_frequency == 0:
                self._save_latent_space()

            if self._r_c.epoch % self._r_c.model_save_frequency == 0:
                self._save_models()

            if self._r_c.epoch % self._r_c.dataset_generation_frequency == 0:
                self._generate_dataset()

        self._generate_dataset()
        self._save_losses()
        self._save_models()
        self._save_samples()
        self._save_latent_space()

        return self._losses

    def resume_training(self, run_folder, checkpoint, new_epochs):
        r_c_filepath = run_folder + 'run_config.json'
        r_c = RunConfig()
        r_c.restore(r_c_filepath)
        r_c.epochs = new_epochs

        m_c_filepath = run_folder + 'model_config.json'
        m_c = BEGANConfig()
        m_c.restore(m_c_filepath)

        self.__init__(r_c, m_c)

        losses_filepath = run_folder + 'losses.p'
        with open(losses_filepath, 'rb') as f:
            self._losses = pickle.load(f)
            for i in range(len(self._losses)):
                self._losses[i] = self._losses[i][:checkpoint]

        self._r_c.epoch = checkpoint

        model_checkpoint = run_folder + 'models/' + str(checkpoint) + '/'
        self.restore_models(model_checkpoint)

    def generate_random_samples(self, n):
        noise = np.random.uniform(-1, 1, (n, self._m_c.latent_dim))
        return self._generator.predict(noise)

    def restore_models(self, models_checkpoint):
        self._generator_model.load_weights(models_checkpoint + 'generator_model.h5')
        self._discriminator_model.load_weights(models_checkpoint + 'discriminator_model.h5')
        self._generator.load_weights(models_checkpoint + 'generator.h5')
        self._discriminator.load_weights(models_checkpoint + 'discriminator.h5')

    def _build_models(self):
        self._generator = began_utils.build_decoder(self._m_c.latent_dim, self._r_c.resolution, self._r_c.channels)
        self._discriminator = began_utils.build_discriminator(self._m_c.latent_dim, self._r_c.resolution,
                                                              self._r_c.channels)
        self._discriminator_model = began_utils.build_discriminator_model(self._discriminator, self._r_c.resolution,
                                                                          self._r_c.channels, self._m_c.initial_lr,
                                                                          self._m_c.loss_exponent)
        self._generator_model = began_utils.build_generator_model(self._generator, self._discriminator,
                                                                  self._m_c.latent_dim, self._m_c.initial_lr,
                                                                  self._m_c.loss_exponent)

    def _update_k(self, discriminator_loss_real, discriminator_loss_generated):
        self._m_c.k = self._m_c.k + self._m_c.lambda_k * (
                    self._m_c.gamma * discriminator_loss_real - discriminator_loss_generated)
        self._m_c.k = np.clip(self._m_c.k, 0.0, 1.0)

    def _save_samples(self):
        noise = np.random.uniform(-1, 1, (self._samples_rows * self._samples_columns, self._m_c.latent_dim))
        generated_samples = self._generator.predict(noise)
        super()._save_samples_common(generated_samples)

    def _save_latent_space(self):
        latent_space_indexes = np.random.choice(self._m_c.latent_dim, 2, replace=False)
        latent_space_inputs = np.zeros((self._latent_grid_size * self._latent_grid_size, self._m_c.latent_dim))

        for i, v_i in enumerate(np.linspace(-1, 1, self._latent_grid_size, True)):
            for j, v_j in enumerate(np.linspace(-1, 1, self._latent_grid_size, True)):
                latent_space_inputs[i * self._latent_grid_size + j, latent_space_indexes] = [v_i, v_j]

        generated_samples = self._generator.predict(latent_space_inputs)
        super()._save_latent_space_common(generated_samples)

    def _save_models(self):
        root_dir = self._r_c.model_dir + '/' + str(self._r_c.epoch) + '/'
        os.makedirs(root_dir)
        self._generator_model.save_weights(root_dir + 'generator_model.h5')
        self._discriminator_model.save_weights(root_dir + 'discriminator_model.h5')
        self._generator.save_weights(root_dir + 'generator.h5')
        self._discriminator.save_weights(root_dir + 'discriminator.h5')

        super()._save_configs()

    def _generate_dataset(self):
        z_samples = np.random.normal(-1, 1, (self._r_c.dataset_generation_size, self._m_c.latent_dim))
        generated_dataset = self._generator.predict(z_samples)

        super()._generate_dataset_common(generated_dataset)

    def get_models(self):
        return self._generator, self._discriminator, self._generator_model, self._discriminator_model
