from functools import partial
from typing import Tuple

from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, Flatten, Lambda
from keras.optimizers import Adam

from emoji_gan.utils.gan_utils import vae_loss, gradient_penalty_loss, \
    RandomWeightedAverage, sampling, set_model_trainable, wasserstein_loss, conv_res_series, deconv_res_series


def build_encoder(latent_dim: int, resolution: int, filters: int = 32, kernel_size: int = 3,
                  channels: int = 3) -> Model:
    image_size = resolution

    encoder_inputs = Input((resolution, resolution, channels))
    encoded = encoder_inputs

    encoded = conv_res_series(encoded, image_size, 4, kernel_size, filters)
    encoded = LeakyReLU()(encoded)

    encoded = Flatten()(encoded)

    z_mean = Dense(latent_dim)(encoded)
    z_log_var = Dense(latent_dim)(encoded)

    encoder = Model(encoder_inputs, [z_mean, z_log_var], name='encoder')
    print(encoder.summary())
    return encoder


def build_decoder(latent_dim: int, resolution: int, filters: int = 32, kernel_size: int = 3,
                  channels: int = 3) -> Model:
    image_size = 4
    filters *= int(resolution / image_size / 2)

    decoder_inputs = Input((latent_dim,))
    decoded = decoder_inputs

    decoded = Dense(image_size * image_size * 8)(decoded)
    decoded = LeakyReLU()(decoded)

    decoded = Reshape((image_size, image_size, 8))(decoded)

    decoded = deconv_res_series(decoded, image_size, resolution, kernel_size, filters)
    decoded = LeakyReLU()(decoded)

    decoded = Conv2D(channels, kernel_size, padding='same', activation='sigmoid')(decoded)

    decoder = Model(decoder_inputs, decoded, name='generator')
    print(decoder.summary())
    return decoder


def build_critic(resolution: int, filters: int = 32, kernel_size: int = 3, channels: int = 3) -> Model:
    image_size = resolution

    critic_inputs = Input((resolution, resolution, channels))
    criticized = critic_inputs

    criticized = conv_res_series(criticized, image_size, 4, kernel_size, filters)
    criticized = LeakyReLU()(criticized)

    criticized = Flatten()(criticized)

    criticized = Dense(1)(criticized)

    critic = Model(critic_inputs, criticized, name='critic')
    print(critic.summary())
    return critic


def build_vae_generator_model(encoder: Model, decoder_generator: Model, critic: Model, latent_dim: int, resolution: int,
                              channels: int, gamma: float, vae_lr: float) -> Tuple[Model, Model]:
    set_model_trainable(encoder, True)
    set_model_trainable(decoder_generator, True)
    set_model_trainable(critic, False)

    real_samples = Input((resolution, resolution, channels))
    noise_samples = Input((latent_dim,))

    generated_samples = decoder_generator(noise_samples)
    generated_criticized = critic(generated_samples)

    z_mean, z_log_var = encoder(real_samples)

    sampled_z = Lambda(sampling)([z_mean, z_log_var])
    decoded_samples = decoder_generator(sampled_z)

    vae_generator_model = Model([real_samples, noise_samples], [generated_criticized, generated_criticized])
    vae_generator_model.compile(optimizer=Adam(lr=vae_lr, beta_1=0.5, beta_2=0.9),
                                loss=[wasserstein_loss, vae_loss(z_mean, z_log_var, real_samples, decoded_samples)],
                                loss_weights=[gamma, (1 - gamma)])

    generator_model = Model(noise_samples, generated_samples, name='generator_model')
    return vae_generator_model, generator_model


def build_critic_model(encoder: Model, decoder_generator: Model, critic: Model, latent_dim: int, resolution: int,
                       batch_size: int, critic_lr: float, gradient_penalty_weight: int, channels: int = 3) -> Model:
    set_model_trainable(encoder, False)
    set_model_trainable(decoder_generator, False)
    set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((resolution, resolution, channels))

    generated_samples = decoder_generator(noise_samples)
    generated_criticized = critic(generated_samples)
    real_criticized = critic(real_samples)

    averaged_samples = RandomWeightedAverage(batch_size)([real_samples, generated_samples])
    averaged_criticized = critic(averaged_samples)

    partial_gp_loss = partial(gradient_penalty_loss,
                              averaged_samples=averaged_samples,
                              gradient_penalty_weight=gradient_penalty_weight)
    partial_gp_loss.__name__ = 'gradient_penalty'

    critic_model = Model([real_samples, noise_samples],
                         [real_criticized, generated_criticized, averaged_criticized], name='critic_model')

    critic_model.compile(optimizer=Adam(critic_lr, beta_1=0.5, beta_2=0.9),
                         loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    return critic_model
