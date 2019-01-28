from functools import partial

from keras import Model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import Adam

from .. import utils


def build_generator(latent_dim, classes_n, resolution, channels, filters=128, kernel_size=3):
    image_size = 4

    latent_input = Input((latent_dim,))
    conditional_input = Input((classes_n,))

    generated = Concatenate()([latent_input, conditional_input])

    generated = Dense(image_size*image_size*32)(generated)
    # generated = BatchNormalization()(generated)
    generated = LeakyReLU()(generated)

    generated = Reshape((image_size, image_size, 32))(generated)

    while image_size != resolution:
        generated = UpSampling2D()(generated)
        generated = Conv2D(filters, kernel_size, padding='same')(generated)
        # generated = BatchNormalization()(generated)
        generated = LeakyReLU()(generated)
        image_size *= 2
        filters = int(filters / 2)

    generated = Conv2D(channels, kernel_size, padding='same', activation='tanh')(generated)

    generator = Model([latent_input, conditional_input], generated, name='generator')
    return generator


def build_critic(resolution, channels, classes_n, filters=32, kernel_size=3):
    image_size = resolution

    critic_inputs = Input((resolution, resolution, channels))
    criticized = critic_inputs

    while image_size != 4:
        criticized = Conv2D(filters, kernel_size, padding='same')(criticized)
        criticized = LeakyReLU()(criticized)
        criticized = MaxPooling2D()(criticized)
        image_size /= 2
        filters = filters * 2

    criticized = Flatten()(criticized)

    class_input = Input((classes_n,))
    criticized = Concatenate()([criticized, class_input])

    criticized = Dense(1)(criticized)

    critic = Model([critic_inputs, class_input], criticized, name='critic')
    return critic


def build_generator_model(generator, critic, latent_dim, classes_n, generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    class_samples = Input((classes_n,))

    generated_samples = generator([noise_samples, class_samples])

    generated_criticized = critic([generated_samples, class_samples])

    generator_model = Model([noise_samples, class_samples], generated_criticized, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9), loss=utils.wasserstein_loss)
    return generator_model


def build_critic_model(generator, critic, latent_dim, resolution, channels, classes_n, batch_size, critic_lr,
                       gradient_penalty_weight):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    class_samples = Input((classes_n,))

    real_samples = Input((resolution, resolution, channels))

    generated_samples = generator([noise_samples, class_samples])
    generated_criticized = critic([generated_samples, class_samples])
    real_criticized = critic([real_samples, class_samples])

    averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
    averaged_criticized = critic([averaged_samples, class_samples])

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penalty_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model([real_samples, noise_samples, class_samples],
                         [real_criticized, generated_criticized, averaged_criticized], name='critic_model')

    critic_model.compile(optimizer=Adam(critic_lr, beta_1=0.5, beta_2=0.9),
                         loss=[utils.wasserstein_loss, utils.wasserstein_loss, partial_gp_loss])
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
