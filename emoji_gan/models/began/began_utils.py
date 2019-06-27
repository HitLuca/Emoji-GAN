import keras.backend as K
from keras import Model, Input
from keras.layers import Dense, Reshape, UpSampling2D, Conv2D, MaxPooling2D, Flatten, Conv2DTranspose, Add, \
    Activation, Lambda, LeakyReLU
from keras.optimizers import Adam

from emoji_gan.utils.gan_utils import deconv_res_series, conv_res_series, set_model_trainable


def build_decoder(latent_dim, resolution, filters=32, kernel_size=3, channels=3):
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

    decoder = Model(decoder_inputs, decoded, name='decoder')
    print(decoder.summary())
    return decoder


def build_encoder(latent_dim, resolution, filters=32, kernel_size=3, channels=3):
    image_size = resolution

    encoder_inputs = Input((resolution, resolution, channels))
    encoded = encoder_inputs

    encoded = conv_res_series(encoded, image_size, 4, kernel_size, filters)
    encoded = LeakyReLU()(encoded)

    encoded = Flatten()(encoded)
    encoded = Dense(latent_dim)(encoded)

    encoder = Model(encoder_inputs, encoded, name='encoder')
    print(encoder.summary())
    return encoder


def build_discriminator(latent_dim, resolution, channels=3):
    input_samples = Input((resolution, resolution, channels))

    encoder = build_encoder(latent_dim, resolution)
    decoder = build_decoder(latent_dim, resolution)

    encoded_samples = encoder(input_samples)
    decoded_samples = decoder(encoded_samples)

    discriminator = Model(input_samples, decoded_samples, name='discriminator')
    print(discriminator.summary())
    return discriminator


def build_generator_model(generator, discriminator, latent_dim, generator_lr, loss_exponent):
    set_model_trainable(discriminator, False)

    input_noise = Input((latent_dim,))
    generated_samples = generator(input_noise)
    discriminated_generated_samples = discriminator(generated_samples)

    ln_generated = ln_loss(generated_samples, discriminated_generated_samples, loss_exponent)
    generator_model = Model(input_noise, ln_generated, name='generator_model')
    generator_model.compile(optimizer=Adam(generator_lr),
                            loss=generator_loss(ln_generated))
    return generator_model


def build_discriminator_model(discriminator, resolution, discriminator_lr, loss_exponent, channels = 3):
    set_model_trainable(discriminator, True)

    real_samples = Input((resolution, resolution, channels))
    generated_samples = Input((resolution, resolution, channels))
    k = Input((1,))

    discriminated_real_samples = discriminator(real_samples)
    discriminated_generated_samples = discriminator(generated_samples)

    ln_real = ln_loss(real_samples, discriminated_real_samples, loss_exponent)
    ln_generated = ln_loss(generated_samples, discriminated_generated_samples, loss_exponent)
    discriminator_model = Model([real_samples, generated_samples, k],
                                [ln_real, ln_generated],
                                name='discriminator_model')

    discriminator_model.compile(optimizer=Adam(discriminator_lr),
                                loss=discriminator_loss(k, ln_real, ln_generated))
    return discriminator_model


def ln_loss(y_true, y_pred, loss_exponent):
    return Lambda(lambda x: K.mean(K.pow(K.abs(x[0] - x[1]), loss_exponent)))([y_true, y_pred])


def generator_loss(ln_generated):
    def loss_function(_, y_pred):
        return ln_generated
    return loss_function


def discriminator_loss(k, ln_real, ln_generated):
    def loss_function(_, y_pred):
        return ln_real - k * ln_generated
    return loss_function
