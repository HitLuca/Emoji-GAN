from functools import partial

from keras import Model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import Adam

from models import utils


def build_generator(latent_dim, resolution, filters=128, kernel_size=4):
    image_size = 4

    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(image_size*image_size*32)(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU()(generated)

    generated = Reshape((image_size, image_size, 32))(generated)

    while image_size != resolution:
        generated = UpSampling2D()(generated)
        generated = Conv2D(filters, kernel_size, padding='same')(generated)
        generated = BatchNormalization()(generated)
        generated = LeakyReLU()(generated)
        image_size *= 2
        filters = int(filters / 2)

    generated_grayscale = Conv2D(1, kernel_size, padding='same', activation='tanh')(generated)

    generated = Concatenate()([generated_grayscale, generated])

    generated = Conv2D(filters, kernel_size, padding='same')(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU()(generated)

    generated = Conv2D(filters, kernel_size, padding='same')(generated)
    generated = BatchNormalization()(generated)
    generated = LeakyReLU()(generated)

    generated_rgb = Conv2D(3, kernel_size, padding='same', activation='tanh')(generated)

    generator = Model(generator_inputs, [generated_grayscale, generated_rgb], 'generator')
    return generator


def build_critic(resolution, filters=32, kernel_size=4):
    image_size = resolution

    critic_inputs_grayscale = Input((resolution, resolution, 1))
    critic_inputs_rgb = Input((resolution, resolution, 3))

    criticized = Concatenate()([critic_inputs_grayscale, critic_inputs_rgb])

    while image_size != 4:
        criticized = Conv2D(filters, kernel_size, padding='same')(criticized)
        criticized = LeakyReLU()(criticized)
        criticized = MaxPooling2D()(criticized)
        image_size /= 2
        filters = filters * 2

    criticized = Flatten()(criticized)

    criticized = Dense(1)(criticized)

    critic = Model([critic_inputs_grayscale, critic_inputs_rgb], criticized, 'critic')

    return critic


def build_generator_model(generator, critic, latent_dim, generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    generated_samples_grayscale, generated_samples_rgb = generator(noise_samples)

    generated_criticized = critic([generated_samples_grayscale, generated_samples_rgb])

    generator_model = Model(noise_samples, generated_criticized, 'generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9), loss=[utils.wasserstein_loss])
    return generator_model


def build_critic_model(generator, critic, latent_dim, resolution, batch_size, critic_lr, gradient_penalty_weight):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples_grayscale = Input((resolution, resolution, 1))
    real_samples_rgb = Input((resolution, resolution, 3))

    generated_samples_grayscale, generated_samples_rgb = generator(noise_samples)

    generated_samples_criticized = critic([generated_samples_grayscale, generated_samples_rgb])
    real_samples_criticized = critic([real_samples_grayscale, real_samples_rgb])

    weights = K.random_uniform((batch_size, 1, 1, 1))
    averaged_samples_grayscale = RandomWeightedAverage(batch_size, weights)([real_samples_grayscale, generated_samples_grayscale])
    averaged_samples_rgb = RandomWeightedAverage(batch_size, weights)([real_samples_rgb, generated_samples_rgb])

    averaged_samples_criticized = critic([averaged_samples_grayscale, averaged_samples_rgb])

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=[averaged_samples_grayscale, averaged_samples_rgb],
                              gradient_penalty_weight=gradient_penalty_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model([real_samples_grayscale, real_samples_rgb, noise_samples],
                         [real_samples_criticized, generated_samples_criticized, averaged_samples_criticized], 'critic_model')

    critic_model.compile(optimizer=Adam(critic_lr, beta_1=0.5, beta_2=0.9),
                         loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss])
    return critic_model


def gradient_penalty_loss(_, y_pred, averaged_samples, gradient_penalty_weight):
    gradients = K.gradients(y_pred, averaged_samples)

    gradient_penalties = []
    for gradient in gradients:
        gradients_sqr = K.square(gradient)
        gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
        gradient_l2_norm = K.sqrt(gradients_sqr_sum)
        gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
        gradient_penalty = K.mean(gradient_penalty)
        gradient_penalties.append(gradient_penalty)
    return 0.5 * (gradient_penalties[0] + gradient_penalties[1])


class RandomWeightedAverage(_Merge):
    def __init__(self, batch_size, weights, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = batch_size
        self._weights = weights

    def _merge_function(self, inputs):
        averaged_inputs = (self._weights * inputs[0]) + ((1 - self._weights) * inputs[1])
        return averaged_inputs
