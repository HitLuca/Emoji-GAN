from functools import partial

from keras import Model
from keras.layers import *
from keras.layers.merge import _Merge
from keras.losses import mean_squared_error
from keras.optimizers import Adam

from models import utils


def build_encoder(latent_dim, resolution, channels, filters=32, kernel_size=3):
    image_size = resolution

    encoder_inputs = Input((resolution, resolution, channels))
    encoded = encoder_inputs

    while image_size != 2:
        encoded = Conv2D(filters, kernel_size, strides=2, padding='same')(encoded)
        encoded = utils.BatchNormalization()(encoded)
        encoded = LeakyReLU(0.2)(encoded)
        image_size /= 2
        filters = filters * 2

    encoded = Conv2D(filters, kernel_size, padding='same')(encoded)
    encoded = LeakyReLU(0.2)(encoded)

    encoded = Flatten()(encoded)

    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)

    encoder = Model(encoder_inputs, [z_mean, z_log_var])
    return encoder


def build_decoder(latent_dim, resolution, channels, filters=256, kernel_size=3):
    image_size = 1

    decoder_inputs = Input((latent_dim,))
    decoded = Reshape((1, 1, latent_dim))(decoder_inputs)

    while image_size != resolution:
        decoded = Conv2DTranspose(filters, kernel_size, strides=2, padding='same')(decoded)
        decoded = utils.BatchNormalization()(decoded)
        decoded = LeakyReLU(0.2)(decoded)
        image_size *= 2
        filters = int(filters / 2)

    decoded = Conv2D(channels, kernel_size, padding='same', activation='tanh')(decoded)

    decoder = Model(decoder_inputs, decoded, 'decoder')
    return decoder


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


def build_vae_model(encoder, decoder_generator, critic, latent_dim, resolution, channels, gamma, vae_lr):
    utils.set_model_trainable(encoder, True)
    utils.set_model_trainable(decoder_generator, True)
    utils.set_model_trainable(critic, False)

    real_samples = Input((resolution, resolution, channels))
    noise_samples = Input((latent_dim,))

    generated_samples = decoder_generator(noise_samples)
    generated_criticized = critic(generated_samples)

    z_mean, z_log_var = encoder(real_samples)

    sampled_z = Lambda(sampling)([z_mean, z_log_var])
    decoded_inputs = decoder_generator(sampled_z)

    real_criticized = critic(real_samples)
    decoded_criticized = critic(decoded_inputs)

    vae_model = Model([real_samples, noise_samples], [generated_criticized, generated_criticized])
    vae_model.compile(optimizer=Adam(lr=vae_lr, beta_1=0, beta_2=0.9),
                      loss=[utils.wasserstein_loss,
                            vae_loss(z_mean, z_log_var, real_criticized, decoded_criticized)],
                      loss_weights=[gamma, (1 - gamma)])

    generator_model = Model(noise_samples, generated_samples)
    return vae_model, generator_model


def vae_loss(z_mean, z_log_var, real_criticized, decoded_criticized):
    def loss(y_true, y_pred):
        mse_loss = mean_squared_error(real_criticized, decoded_criticized)
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(mse_loss + kl_loss)

    return loss


def build_critic_model(encoder, decoder_generator, critic, latent_dim, resolution, channels, batch_size, critic_lr,
                       gradient_penality_weight):
    utils.set_model_trainable(encoder, False)
    utils.set_model_trainable(decoder_generator, False)
    utils.set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((resolution, resolution, channels))

    generated_samples = decoder_generator(noise_samples)
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


def sampling(args):
    z_mean, z_log_var = args
    batch_size = K.shape(z_mean)[0]
    latent_dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch_size, latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon