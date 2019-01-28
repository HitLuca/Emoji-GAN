from keras import Model, Input
from keras.layers import Dense, LeakyReLU, Reshape, UpSampling2D, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.optimizers import Adam
from models import utils


def build_generator(latent_dim, resolution, channels, filters=128, kernel_size=3):
    image_size = 4

    generator_inputs = Input((latent_dim,))
    generated = generator_inputs

    generated = Dense(image_size * image_size * 32)(generated)
    # generated = BatchNormalization()(generated)
    generated = LeakyReLU(0.2)(generated)

    generated = Reshape((image_size, image_size, 32))(generated)

    while image_size != resolution:
        generated = UpSampling2D()(generated)
        generated = Conv2D(filters, kernel_size, padding='same')(generated)
        # generated = BatchNormalization()(generated)
        generated = LeakyReLU(0.2)(generated)
        image_size *= 2
        filters = int(filters / 2)

    generated = Conv2D(channels, kernel_size, padding='same', activation='tanh')(generated)

    generator = Model(generator_inputs, generated, name='generator')
    return generator


def build_discriminator(resolution, channels, filters=32, kernel_size=3):
    image_size = resolution

    discriminator_inputs = Input((resolution, resolution, channels))
    discriminated = discriminator_inputs

    while image_size != 4:
        discriminated = Conv2D(filters, kernel_size, padding='same')(discriminated)
        discriminated = LeakyReLU(0.2)(discriminated)
        discriminated = MaxPooling2D()(discriminated)
        image_size /= 2
        filters = filters * 2

    discriminated = Flatten()(discriminated)

    discriminated = Dense(1, activation='sigmoid')(discriminated)

    discriminator = Model(discriminator_inputs, discriminated, name='discriminator')

    return discriminator


def build_generator_model(generator, discriminator, latent_dim, generator_lr):
    utils.set_model_trainable(generator, True)
    utils.set_model_trainable(discriminator, False)

    noise_samples = Input((latent_dim,))
    generated_samples = generator(noise_samples)

    generated_discriminated = discriminator(generated_samples)

    generator_model = Model([noise_samples], generated_discriminated, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr, beta_1=0.5, beta_2=0.9), loss='binary_crossentropy')
    return generator_model


def build_discriminator_model(generator, discriminator, latent_dim, resolution, channels, discriminator_lr):
    utils.set_model_trainable(generator, False)
    utils.set_model_trainable(discriminator, True)

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
