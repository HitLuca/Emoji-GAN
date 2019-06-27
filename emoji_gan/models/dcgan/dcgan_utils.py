from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, Conv2D, Flatten
from keras.optimizers import Adam

from emoji_gan.utils.gan_utils import deconv_series, conv_series, set_model_trainable


def build_generator(latent_dim: int, resolution: int, filters: int = 32, kernel_size: int = 3,
                    channels: int = 3) -> Model:
    image_size = 4

    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(image_size * image_size * 8)(generated)
    generated = LeakyReLU()(generated)

    generated = Reshape((image_size, image_size, 8))(generated)

    generated = deconv_series(generated, image_size, resolution, kernel_size, filters)

    generated = Conv2D(channels, kernel_size, padding='same', activation='sigmoid')(generated)

    generator = Model(generator_inputs, generated, name='generator')
    return generator


def build_discriminator(resolution: int, filters: int = 32, kernel_size: int = 3, channels: int = 3) -> Model:
    image_size = resolution

    discriminator_inputs = Input((resolution, resolution, channels))
    discriminated = discriminator_inputs

    discriminated = conv_series(discriminated, image_size, 4, kernel_size, filters)

    discriminated = Flatten()(discriminated)

    discriminated = Dense(1, activation='sigmoid')(discriminated)

    critic = Model(discriminator_inputs, discriminated, name='discriminator')
    return critic


def build_generator_model(generator: Model, discriminator: Model, latent_dim: int, generator_lr: float) -> Model:
    set_model_trainable(generator, True)
    set_model_trainable(discriminator, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    generated_discriminated = discriminator(generated_samples)

    generator_model = Model(noise_samples, generated_discriminated, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9), loss='binary_crossentropy')
    return generator_model


def build_discriminator_model(generator: Model, discriminator: Model, latent_dim: int, resolution: int,
                              discriminator_lr: float, channels: int = 3) -> Model:
    set_model_trainable(generator, False)
    set_model_trainable(discriminator, True)

    noise_samples = Input((latent_dim,))
    real_samples = Input((resolution, resolution, channels))

    generated_samples = generator(noise_samples)
    generated_discriminated = discriminator(generated_samples)
    real_discriminated = discriminator(real_samples)

    discriminator_model = Model([real_samples, noise_samples],
                                [real_discriminated, generated_discriminated], name='discriminator_model')

    discriminator_model.compile(optimizer=Adam(discriminator_lr, beta_1=0.5, beta_2=0.9),
                                loss=['binary_crossentropy', 'binary_crossentropy'])
    return discriminator_model
