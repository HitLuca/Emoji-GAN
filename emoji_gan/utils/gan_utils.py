from typing import List

import numpy as np
from keras import backend as K, Model
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU, UpSampling2D, Add
from keras.layers.merge import _Merge


def set_model_trainable(model: Model, trainable: bool) -> Model:
    model.trainable = trainable
    for l in model.layers:
        l.trainable = trainable
    return model


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)


def apply_lr_decay(models: List[Model], lr_decay_factor: float) -> None:
    for model in models:
        lr_tensor = model.optimizer.lr
        lr = K.get_value(lr_tensor)
        K.set_value(lr_tensor, lr * lr_decay_factor)


def conv_block(inputs, kernel_size: int, filters: int):
    block = Conv2D(filters, kernel_size, padding='same')(inputs)
    block = LeakyReLU()(block)
    block = MaxPooling2D()(block)
    return block


def deconv_block(inputs, kernel_size: int, filters: int):
    block = UpSampling2D()(inputs)
    block = Conv2D(filters, kernel_size, padding='same')(block)
    block = LeakyReLU()(block)
    return block


def conv_res_block(inputs, kernel_size: int, filters: int, pooling: bool = False):
    block = inputs
    res = inputs

    block = LeakyReLU()(block)
    block = Conv2D(filters, kernel_size, padding='same')(block)

    block = LeakyReLU()(block)
    block = Conv2D(filters, kernel_size, padding='same')(block)

    res = Conv2D(filters, kernel_size=1)(res)
    if pooling:
        res = MaxPooling2D()(res)
        block = MaxPooling2D()(block)

    block = Add()([res, block])
    return block


def deconv_res_block(inputs, kernel_size: int, filters: int, upsampling: bool = False):
    block = inputs
    res = inputs

    if upsampling:
        block = UpSampling2D()(block)
        res = UpSampling2D()(res)

    block = LeakyReLU()(block)
    block = Conv2D(filters, kernel_size, padding='same')(block)

    block = LeakyReLU()(block)
    block = Conv2D(filters, kernel_size, padding='same')(block)

    res = Conv2D(filters, kernel_size=1)(res)
    block = Add()([res, block])
    return block


def conv_res_series(inputs, image_size: int, target_resolution: int, kernel_size: int, filters: int):
    block = inputs
    while image_size != target_resolution:
        block = conv_res_block(block, kernel_size, filters, pooling=True)

        image_size /= 2
        filters *= 2

    return block


def deconv_res_series(inputs, image_size: int, target_resolution: int, kernel_size: int, filters: int):
    block = inputs
    while image_size != target_resolution:
        block = deconv_res_block(block, kernel_size, filters, upsampling=True)

        image_size *= 2
        filters = int(filters / 2)

    return block


def conv_series(inputs, image_size: int, target_resolution: int, kernel_size: int, filters: int):
    block = inputs
    while image_size != target_resolution:
        block = conv_block(block, kernel_size, filters)

        image_size /= 2
        filters = filters * 2

    return block


def deconv_series(inputs, image_size: int, target_resolution: int, kernel_size: int, filters: int):
    block = inputs
    while image_size != target_resolution:
        block = deconv_block(block, kernel_size, filters)

        image_size *= 2
        filters = int(filters / 2)

    return block


def vae_loss(z_mean, z_log_var, real, predicted):
    def loss(y_true, y_pred):
        mse_loss = K.mean(K.binary_crossentropy(real, predicted))
        kl_loss = - 0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return K.mean(mse_loss + kl_loss)

    return loss


def gradient_penalty_loss(_, y_pred, averaged_samples, gradient_penalty_weight: int):
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
