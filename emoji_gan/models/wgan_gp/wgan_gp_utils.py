from functools import partial

from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, Flatten, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.utils import plot_model

from emoji_gan.utils.gan_utils import conv_series, deconv_series, gradient_penalty_loss, \
    RandomWeightedAverage, set_model_trainable, deconv_res_series, conv_res_series

from emoji_gan.utils.gan_utils import wasserstein_loss


def build_generator(latent_dim, resolution, filters=32, kernel_size=3, channels=3):
    image_size = 4
    filters *= int(resolution / image_size / 2)

    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(image_size * image_size * 8)(generated)
    generated = LeakyReLU()(generated)

    generated = Reshape((image_size, image_size, 8))(generated)

    generated = deconv_res_series(generated, image_size, resolution, kernel_size, filters)
    generated = LeakyReLU()(generated)

    generated = Conv2D(channels, kernel_size, padding='same', activation='sigmoid')(generated)

    generator = Model(generator_inputs, generated, name='generator')
    print(generator.summary())
    return generator


def build_critic(resolution, filters=32, kernel_size=3, channels=3):
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


def build_generator_model(generator, critic, latent_dim, generator_lr):
    set_model_trainable(generator, True)
    set_model_trainable(critic, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    generated_criticized = critic(generated_samples)

    generator_model = Model(noise_samples, generated_criticized, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9),
                            loss=wasserstein_loss)
    return generator_model


def build_critic_model(generator, critic, latent_dim, resolution, channels, batch_size, critic_lr,
                       gradient_penalty_weight):
    set_model_trainable(generator, False)
    set_model_trainable(critic, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((resolution, resolution, channels))

    generated_samples = generator(noise_samples)
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
