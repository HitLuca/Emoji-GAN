import os
import pickle

from keras.layers import *

from models import utils
from models.ds_wgan_gp import ds_wgan_gp_utils


class DS_WGAN_GP:
    def __init__(self, config):
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

        self._epoch = 0
        self._losses = [[], []]
        self._build_models()

        os.makedirs(self._img_dir + '/grayscale')
        os.makedirs(self._img_dir + '/rgb')
        os.makedirs(self._generated_datasets_dir + '/grayscale')
        os.makedirs(self._generated_datasets_dir + '/rgb')

    def _build_models(self):
        self._generator = ds_wgan_gp_utils.build_generator(self._latent_dim, self._resolution)

        self._critic = ds_wgan_gp_utils.build_critic(self._resolution)

        self._generator_model = ds_wgan_gp_utils.build_generator_model(self._generator, self._critic,
                                                                       self._latent_dim, self._generator_lr)
        self._critic_model = ds_wgan_gp_utils.build_critic_model(self._generator, self._critic,
                                                                 self._latent_dim, self._resolution, self._batch_size,
                                                                 self._critic_lr, self._gradient_penalty_weight)

    def train(self, dataset_grayscale, dataset_rgb, *_):
        ones = np.ones((self._batch_size, 1))
        neg_ones = -ones
        zeros = np.zeros((self._batch_size, 1))

        while self._epoch < self._epochs:
            self._epoch += 1
            critic_losses = []
            for _ in range(self._n_critic):
                indexes = np.random.randint(0, dataset_grayscale.shape[0], self._batch_size)
                real_samples_grayscale = dataset_grayscale[indexes]
                real_samples_rgb = dataset_rgb[indexes]
                noise = np.random.normal(0, 1, (self._batch_size, self._latent_dim))
                inputs = [real_samples_grayscale, real_samples_rgb, noise]

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
        generated_samples_grayscale, generated_samples_rgb = self._generator.predict(noise)

        filenames_grayscale = [self._img_dir + ('/grayscale/%07d.png' % self._epoch), self._img_dir + '/grayscale/last.png']
        filenames_rgb = [self._img_dir + ('/rgb/%07d.png' % self._epoch), self._img_dir + '/rgb/last.png']
        utils.save_samples(generated_samples_grayscale, rows, columns, self._resolution, 1, filenames_grayscale)
        utils.save_samples(generated_samples_rgb, rows, columns, self._resolution, 3, filenames_rgb)

    def _save_latent_space(self):
        grid_size = 6

        latent_space_inputs = np.zeros((grid_size * grid_size, self._latent_dim))

        for i, v_i in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
            for j, v_j in enumerate(np.linspace(-1.5, 1.5, grid_size, True)):
                latent_space_inputs[i * grid_size + j, :2] = [v_i, v_j]

        generated_samples_grayscale, generated_samples_rgb = self._generator.predict(latent_space_inputs)

        filenames = [self._img_dir + '/grayscale/latent_space.png', self._img_dir + ('/grayscale/%07d_latent_space.png' % self._epoch)]
        utils.save_latent_space(generated_samples_grayscale, grid_size, self._resolution, 1, filenames)

        filenames = [self._img_dir + '/rgb/latent_space.png', self._img_dir + ('/rgb/%07d_latent_space.png' % self._epoch)]
        utils.save_latent_space(generated_samples_rgb, grid_size, self._resolution, 3, filenames)

    def _save_losses(self):
        utils.save_losses_wgan(self._losses, self._img_dir + '/losses.png')

        with open(self._run_dir + '/losses.p', 'wb') as f:
            pickle.dump(self._losses, f)

    def _save_models(self):
        root_dir = self._model_dir + '/' + str(self._epoch) + '/'
        os.makedirs(root_dir)
        self._generator_model.save(root_dir + 'generator_model.h5')
        self._critic_model.save(root_dir + 'critic_model.h5')
        self._generator.save(root_dir + 'generator.h5')
        self._critic.save(root_dir + 'critic.h5')

    def _generate_dataset(self):
        z_samples = np.random.normal(0, 1, (self._dataset_generation_size, self._latent_dim))
        generated_dataset_grayscale, generated_dataset_rgb = self._generator.predict(z_samples)
        np.save(self._generated_datasets_dir + ('/grayscale/%d_generated_data' % self._epoch), generated_dataset_grayscale)
        np.save(self._generated_datasets_dir + ('/rgb/%d_generated_data' % self._epoch), generated_dataset_rgb)

    def get_models(self):
        return self._generator, self._critic, self._generator_model, self._critic_model

    def _apply_lr_decay(self):
        lr_tensor = self._generator_model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * self._lr_decay_factor)

        lr_tensor = self._critic_model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * self._lr_decay_factor)
