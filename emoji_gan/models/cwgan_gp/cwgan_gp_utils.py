from functools import partial

from keras import Model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.optimizers import Adam

from .. import utils


def build_generator(latent_dim, classes_n, resolution, channels, filters=256, kernel_size=3):
    image_size = 4

    latent_input = Input((latent_dim,))
    conditional_input = Input((classes_n,))

    generator_inputs = Concatenate()([latent_input, conditional_input])
    generated = generator_inputs

    generated = Dense(image_size*image_size*32)(generated)
    generated = LeakyReLU(0.2)(generated)

    generated = Reshape((image_size, image_size, 32))(generated)

    while image_size != resolution/2:
        generated = UpSampling2D()(generated)
        generated = Conv2D(filters, kernel_size, padding='same')(generated)
        generated = LeakyReLU(0.2)(generated)
        image_size *= 2
        filters = int(filters / 2)

    generated = UpSampling2D()(generated)
    generated = Conv2D(channels, kernel_size, padding='same', activation='tanh')(generated)

    generator = Model([generator_inputs, conditional_input], generated, 'generator')
    return generator


def build_critic(resolution, channels, classes_n, filters=32, kernel_size=3):
    image_size = resolution

    critic_inputs = Input((resolution, resolution, channels))
    criticized = critic_inputs

    while image_size != 4:
        criticized = Conv2D(filters, kernel_size, padding='same')(criticized)
        criticized = LeakyReLU(0.2)(criticized)
        criticized = MaxPooling2D()(criticized)
        image_size /= 2
        filters = filters * 2

    class_input = Input((classes_n,))
    criticized = Concatenate()([criticized, class_input])

    criticized = Dense(128)(criticized)
    criticized = LeakyReLU(0.2)(criticized)

    criticized = Dense(1)(criticized)
    critic = Model([critic_inputs, class_input], criticized, 'critic')
    return critic


def build_generator_model(generator, critic, latent_dim, classes_n, generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    class_samples = Input((classes_n,))

    generated_samples = generator([noise_samples, class_samples])

    generated_criticized = critic([generated_samples, class_samples])

    generator_model = Model([noise_samples, class_samples], generated_criticized, 'generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0, beta_2=0.9), loss=utils.wasserstein_loss)
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
