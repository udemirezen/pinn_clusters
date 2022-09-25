from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Initializer

from formulations.constants import *

class TunalbeXavierNormal(Initializer):

    def __init__(self, lsnow: float):
        self._lsnow = float(lsnow)

    def __call__(self, shape, dtype, **kwargs):
        xavier_stddev = tf.sqrt(2 / tf.math.reduce_sum(shape)) * self._lsnow
        return tf.Variable(tf.random.truncated_normal(shape, stddev=xavier_stddev, dtype=dtype))

    def get_config(self):
        return {"lsnow": self._lsnow}


class ShiftLayer(Layer):
    """A layer for shifting the input by pre-defined constant ux, lx"""

    def call(self, inputs):
        return 2.0 * (inputs - lx) / (ux - lx) - 1.0


def create_mlp(layers: List[int], lyscl: List[int], dtype=tf.float64):
    """Multilayer perceptron for PINN problem"""

    inputs = Input(shape=(1,), dtype=dtype)
    shifted = ShiftLayer()(inputs)
    dense = Dense(
        layers[0], activation="tanh", dtype=dtype,
        kernel_initializer=TunalbeXavierNormal(lyscl[0]))(shifted)
    for n_unit, stddev in zip(layers[1:-1], lyscl[1:-1]):
        dense = Dense(
            n_unit, activation="tanh", dtype=dtype,
            kernel_initializer=TunalbeXavierNormal(stddev))(dense)
    dense = Dense(
        layers[-1], activation=None, dtype=dtype,
        kernel_initializer=TunalbeXavierNormal(lyscl[-1]))(dense)
    model = Model(inputs=inputs, outputs=dense)
    return model
