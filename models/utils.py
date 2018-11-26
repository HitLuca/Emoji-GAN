import json
import os
from datetime import datetime

import keras.backend as K
import numpy as np
from keras import initializers, regularizers, constraints
from keras.engine import Layer, InputSpec
# import matplotlib
# matplotlib.use('Agg')
from keras.legacy import interfaces
from matplotlib import pyplot as plt


def load_unlabelled_dataset(company, resolution, channels):
    unlabelled_dataset_filepath = '../../dataset/'
    dataset = np.load(unlabelled_dataset_filepath + company + '_' + str(resolution) + '.npy')
    return np.reshape(dataset, (dataset.shape[0] * dataset.shape[1], resolution, resolution, channels))


def load_labelled_dataset(company, resolution, channels):
    unlabelled_dataset_filepath = '../../dataset/'
    dataset = np.load(unlabelled_dataset_filepath + company + '_' + str(resolution) + '.npy')
    dataset = dataset[:, :, :, :channels]
    return dataset


def set_model_trainable(model, trainable):
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable


def save_samples(generated_data, rows, columns, resolution, channels, filenames):
    plt.subplots(rows, columns, figsize=(7, 7))

    k = 1
    for i in range(rows):
        for j in range(columns):
            plt.subplot(rows, columns, k)
            plt.imshow((generated_data[k - 1].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            plt.xticks([])
            plt.yticks([])
            k += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def save_samples_classes(generated_data, labels, rows, columns, resolution, channels, filenames):
    plt.subplots(rows, columns, figsize=(7, 7))

    k = 1
    for i in range(rows):
        for j in range(columns):
            plt.subplot(rows, columns, k)
            plt.title(labels[k - 1])
            plt.imshow((generated_data[k - 1].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            plt.xticks([])
            plt.yticks([])
            k += 1
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.5)
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def save_losses_wgan(losses, filename, legend_name='critic'):
    plt.figure(figsize=(15, 4.5))
    plt.plot(losses[0])
    plt.plot(losses[1])
    plt.legend(['generator', legend_name])
    plt.savefig(filename)
    plt.clf()
    plt.close()


def save_losses_other(losses, filename, text):
    plt.figure(figsize=(15, 4.5))
    plt.plot(losses)
    plt.legend([text])
    plt.savefig(filename)
    plt.clf()
    plt.close()


def save_losses_wgan_gp_ae(losses, filename, legend_name='generator AE'):
    plt.subplots(2, 1, figsize=(15, 9))
    plt.subplot(2, 1, 1)
    plt.plot(losses[0])
    plt.plot(losses[1])
    plt.legend(['generator', 'critic'])

    plt.subplot(2, 1, 2)
    plt.plot(losses[2])
    plt.legend([legend_name])

    plt.savefig(filename)
    plt.clf()
    plt.close()


def save_latent_space(generated_data, grid_size, resolution, channels, filenames):
    plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))

    for i in range(grid_size):
        for j in range(grid_size):
            plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
            plt.imshow((generated_data[i * grid_size + j].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            plt.xticks([])
            plt.yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def save_latent_space_classes(generated_data, chosen_class, grid_size, resolution, channels, filenames):
    plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    plt.suptitle(chosen_class)

    for i in range(grid_size):
        for j in range(grid_size):
            plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
            plt.imshow((generated_data[i * grid_size + j].reshape(resolution, resolution, channels) + 1.0) / 2.0)
            plt.xticks([])
            plt.yticks([])
    for filename in filenames:
        plt.savefig(filename)
    plt.clf()
    plt.close()


def generate_run_dir():
    root_path = 'outputs'
    if not os.path.exists(root_path):
        os.mkdir(root_path)

    current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_dir = root_path + '/' + current_datetime
    img_dir = run_dir + '/img'
    model_dir = run_dir + '/models'
    generated_datesets_dir = run_dir + '/generated_datasets'

    os.mkdir(run_dir)
    os.mkdir(img_dir)
    os.mkdir(model_dir)
    os.mkdir(generated_datesets_dir)

    return run_dir, img_dir, model_dir, generated_datesets_dir


def get_global_config():
    batch_size = 64
    epochs = 1000000
    latent_dim = 10
    img_frequency = 250
    loss_frequency = 125
    latent_space_frequency = 1000
    model_save_frequency = 1000
    dataset_generation_frequency = 1000
    dataset_generation_size = 1000

    n_generator = 1
    n_critic = 5
    generator_lr = 0.001
    critic_lr = 0.001
    gradient_penality_weight = 10
    gamma = 0.5

    lr_decay_factor = 0.5
    lr_decay_steps = epochs / 4

    config = {
        'batch_size': batch_size,
        'epochs': epochs,
        'latent_dim': latent_dim,
        'img_frequency': img_frequency,
        'loss_frequency': loss_frequency,
        'latent_space_frequency': latent_space_frequency,
        'model_save_frequency': model_save_frequency,
        'dataset_generation_frequency': dataset_generation_frequency,
        'dataset_generation_size': dataset_generation_size,
        'lr_decay_factor': lr_decay_factor,
        'lr_decay_steps': lr_decay_steps,
        'n_generator': n_generator,
        'n_critic': n_critic,
        'gradient_penality_weight': gradient_penality_weight,
        'gamma': gamma,
        'generator_lr': generator_lr,
        'critic_lr': critic_lr
    }

    return config


def merge_config_and_save(config_2):
    config = get_global_config()
    config.update(config_2)

    with open(config['run_dir'] + '/config.json', 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    return config


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


class BatchNormalization(Layer):
    """Batch normalization layer (Ioffe and Szegedy, 2014).
    Normalize the activations of the previous layer at each batch,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `BatchNormalization`.
        momentum: Momentum for the moving mean and the moving variance.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        moving_mean_initializer: Initializer for the moving mean.
        moving_variance_initializer: Initializer for the moving variance.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
    """

    @interfaces.legacy_batchnorm_support
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        self._trainable = True
        self._trainable_tensor = K.variable(1, dtype='float32', name='trainable')
        super(BatchNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.momentum = momentum
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.moving_mean_initializer = initializers.get(moving_mean_initializer)
        self.moving_variance_initializer = initializers.get(moving_variance_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    @property
    def trainable(self):
        # Use cached value to avoid unnecessary get_value() calls
        return self._trainable

    @trainable.setter
    def trainable(self, trainable):
        trainable = bool(trainable)
        # Change when different to avoid unnecessary set_value() calls
        if self._trainable != trainable:
            self._trainable = trainable
            K.set_value(self._trainable_tensor, 1 if trainable else 0)

    def build(self, input_shape):
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError('Axis ' + str(self.axis) + ' of '
                                                        'input tensor should have a defined dimension '
                                                        'but the layer received an input with shape ' +
                             str(input_shape) + '.')
        self.input_spec = InputSpec(ndim=len(input_shape),
                                    axes={self.axis: dim})
        shape = (dim,)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.moving_mean = self.add_weight(
            shape=shape,
            name='moving_mean',
            initializer=self.moving_mean_initializer,
            trainable=False)
        self.moving_variance = self.add_weight(
            shape=shape,
            name='moving_variance',
            initializer=self.moving_variance_initializer,
            trainable=False)
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        # Prepare broadcasting shape.
        ndim = len(input_shape)
        reduction_axes = list(range(len(input_shape)))
        del reduction_axes[self.axis]
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        # Determines whether broadcasting is needed.
        needs_broadcasting = (sorted(reduction_axes) != list(range(ndim))[:-1])

        def normalize_inference():
            if needs_broadcasting:
                # In this case we must explicitly broadcast all parameters.
                broadcast_moving_mean = K.reshape(self.moving_mean,
                                                  broadcast_shape)
                broadcast_moving_variance = K.reshape(self.moving_variance,
                                                      broadcast_shape)
                if self.center:
                    broadcast_beta = K.reshape(self.beta, broadcast_shape)
                else:
                    broadcast_beta = None
                if self.scale:
                    broadcast_gamma = K.reshape(self.gamma,
                                                broadcast_shape)
                else:
                    broadcast_gamma = None
                return K.batch_normalization(
                    inputs,
                    broadcast_moving_mean,
                    broadcast_moving_variance,
                    broadcast_beta,
                    broadcast_gamma,
                    epsilon=self.epsilon)
            else:
                return K.batch_normalization(
                    inputs,
                    self.moving_mean,
                    self.moving_variance,
                    self.beta,
                    self.gamma,
                    epsilon=self.epsilon)

        if training in {0, False}:
            # If the learning phase is *static* and set to inference:
            return normalize_inference()
        elif training is None:
            # If it's undefined then if trainable tensor is on respect learning phase else set to false
            training = K.switch(self._trainable_tensor, K.cast(K.learning_phase(), 'float32'),
                                K.constant(0, dtype='float32'))
            training._uses_learning_phase = True

        # If the learning is either dynamic, or set to training:
        normed_training, mean, variance = K.normalize_batch_in_training(
            inputs, self.gamma, self.beta, reduction_axes,
            epsilon=self.epsilon)

        if K.backend() != 'cntk':
            sample_size = K.prod([K.shape(inputs)[axis]
                                  for axis in reduction_axes])
            sample_size = K.cast(sample_size, dtype=K.dtype(inputs))

            # sample variance - unbiased estimator of population variance
            variance *= sample_size / (sample_size - (1.0 + self.epsilon))

        self.add_update([K.moving_average_update(self.moving_mean,
                                                 mean,
                                                 self.momentum),
                         K.moving_average_update(self.moving_variance,
                                                 variance,
                                                 self.momentum)],
                        inputs)

        # Pick the normalized form corresponding to the training phase.
        return K.in_train_phase(normed_training,
                                normalize_inference,
                                training=training)

    def get_config(self):
        config = {
            'axis': self.axis,
            'momentum': self.momentum,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'moving_mean_initializer': initializers.serialize(self.moving_mean_initializer),
            'moving_variance_initializer': initializers.serialize(self.moving_variance_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(BatchNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
