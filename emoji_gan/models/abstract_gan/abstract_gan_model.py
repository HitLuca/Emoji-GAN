import abc
import pickle
import numpy as np

from models import utils


class AbstractGAN(abc.ABC):
    def __init__(self, r_c, m_c):
        self._r_c = r_c
        self._m_c = m_c
        self._losses = []
        self._legend_names = []

        self._samples_rows, self._samples_columns = 6, 6
        self._latent_grid_size = 6

    @abc.abstractmethod
    def train(self, dataset, classes):
        pass

    @abc.abstractmethod
    def resume_training(self, run_filepath, checkpoint, new_epochs):
        pass

    @abc.abstractmethod
    def generate_random_samples(self, n):
        pass

    @abc.abstractmethod
    def restore_models(self, models_checkpoint):
        pass

    @abc.abstractmethod
    def _build_models(self):
        pass

    @abc.abstractmethod
    def _save_samples(self):
        pass

    def _save_samples_common(self, generated_samples):
        filenames = [self._r_c.img_dir + ('/%07d.png' % self._r_c.epoch), self._r_c.img_dir + '/last.png']
        utils.save_samples(generated_samples, self._samples_rows, self._samples_columns,
                           self._r_c.resolution, self._r_c.channels, filenames)

    @abc.abstractmethod
    def _save_latent_space(self):
        pass

    @abc.abstractmethod
    def _save_models(self):
        pass

    @abc.abstractmethod
    def _generate_dataset(self):
        pass

    @abc.abstractmethod
    def get_models(self):
        pass

    def _save_losses(self):
        utils.save_losses(self._losses, self._r_c.img_dir + '/losses.png', self._legend_names)

        with open(self._r_c.run_dir + '/losses.p', 'wb') as f:
            pickle.dump(self._losses, f)

    def _generate_dataset_common(self, generated_dataset):
        np.save(self._r_c.generated_datasets_dir + ('/%d_generated_data' % self._r_c.epoch), generated_dataset)

    def _save_latent_space_common(self, generated_samples):
        filenames = [self._r_c.img_dir + '/latent_space.png',
                     self._r_c.img_dir + ('/%07d_latent_space.png' % self._r_c.epoch)]
        utils.save_latent_space(generated_samples, self._latent_grid_size,
                                self._r_c.resolution, self._r_c.channels, filenames)

    def _save_configs(self):
        r_c_filepath = self._r_c.run_dir + '/run_config.json'
        self._r_c.save(r_c_filepath)

        m_c_filepath = self._r_c.run_dir + '/model_config.json'
        self._m_c.save(m_c_filepath)
