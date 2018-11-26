from functools import partial

from keras import Model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import Adam

from models import utils


def build_generator(latent_dim, resolution, channels, filters=256, kernel_size=3):
    image_size = 1

    generator_inputs = Input((latent_dim,))
    generated = Reshape((1, 1, latent_dim))(generator_inputs)

    while image_size != resolution:
        generated = Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(generated)
        generated = utils.BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        image_size *= 2
        filters = int(filters / 2)

    generated = Conv2D(channels, kernel_size, padding='same', activation='tanh')(generated)

    generator = Model(generator_inputs, generated, 'generator')
    return generator


def build_critic(resolution, channels, filters=32, kernel_size=3):
    image_size = resolution

    critic_inputs = Input((resolution, resolution, channels))
    criticized = critic_inputs

    while image_size != 2:
        criticized = Conv2D(filters, kernel_size, strides=2, padding='same')(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        image_size /= 2
        filters = filters * 2

    criticized = Conv2D(filters, kernel_size, padding='same')(criticized)
    criticized = LeakyReLU(0.2)(criticized)

    criticized = Flatten()(criticized)

    criticized = Dense(50)(criticized)
    criticized = LeakyReLU(0.2)(criticized)
    criticized = Dense(1)(criticized)

    critic = Model(critic_inputs, criticized, 'critic')

    return critic


def build_generator_model(generator, critic, latent_dim, generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    generated_criticized = critic(generated_samples)

    generator_model = Model([noise_samples], generated_criticized, 'generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0, beta_2=0.9), loss=utils.wasserstein_loss)
    return generator_model


def build_critic_model(generator, critic, latent_dim, resolution, channels, batch_size, critic_lr,
                       gradient_penality_weight):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((resolution, resolution, channels))

    generated_samples = generator(noise_samples)
    generated_criticized = critic(generated_samples)
    real_criticized = critic(real_samples)

    averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
    averaged_criticized = critic(averaged_samples)

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penality_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model([real_samples, noise_samples],
                         [real_criticized, generated_criticized, averaged_criticized], 'critic_model')

    critic_model.compile(optimizer=Adam(critic_lr, beta_1=0, beta_2=0.9),
                         loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss],
                         loss_weights=[1 / 3, 1 / 3, 1 / 3])
    return critic_model


def gradient_penalty_loss(_, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)[0]
    gradients_sqr = K.square(gradients)
    gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    return K.mean(gradient_penalty)


class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size

    def _merge_function(self, inputs):
        weights = K.random_uniform((self._batch_size, 1, 1, 1))
        averaged_inputs = (weights * inputs[0]) + ((1 - weights) * inputs[1])
        return averaged_inputs
