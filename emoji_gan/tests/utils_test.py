import pytest
from keras import Input, Model
from keras.layers import Dense
from emoji_gan.models import utils


@pytest.fixture
def test_model():
    inputs = Input((1,))
    outputs = Dense(1)(inputs)
    model = Model(inputs, outputs)
    model.compile(loss='mse', optimizer='sgd')
    return model


def test_set_model_trainable(test_model):
    utils.set_model_trainable(test_model, False)

    layers_trainable = False

    for layer in test_model.layers:
        if layer.trainable:
            layers_trainable = True

    assert not test_model.trainable and not layers_trainable
