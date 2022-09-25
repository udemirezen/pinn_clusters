import numpy as np
from pyDOE import lhs
import tensorflow as tf

from formulations.constants import *

_data_type = tf.float64

def to_mat_tensor(var):
    return tf.constant([var], dtype=_data_type)[None, :]

def to_tensor(var):
    return tf.constant(var, dtype=_data_type)

# Stores 2 important analytic results for testing
def analytic_h_constantB(x):
    return ((A0 * h0 ** (n + 1) * (A0 * x + q0) ** (n + 1)) / (
            A0 * q0 ** (n + 1) - (q0 * h0) ** (n + 1) + (h0 * (A0 * x + q0)) ** (n + 1))) ** (1 / (n + 1))

def analytic_u_constantB(x):
    return (A0 * x + q0) / analytic_h_constantB(x)


def get_collocation_points(x_train, xmin: float, xmax: float, N_t: int):
    # random points around the center
    collocation_pts = xmin + (xmax - xmin) * lhs(1, N_t)
    #focused_pts = xmin + 0.05*lhs(1,1000)
    #collocation_pts = np.vstack((collocation_pts,focused_pts))
    #change to a stateful random generator.
    ##collocation_pts = xmin + (xmax - xmin) *tf.random.uniform((N_t,1))
    # biases towards zero
    collocation_pts = collocation_pts ** 3
    # combines all x points used so far
    #remove bias
    ###collocation_pts = np.vstack((collocation_pts, x_train))
    #replaced with tf.concat, because the collocation_pts is now a tensorflow Tensor, incompatible with np. vstack
    ##collocation_pts = tf.concat([collocation_pts, x_train],axis = 0)
    # return as tensorflow array
    return tf.cast(collocation_pts, dtype=_data_type)
