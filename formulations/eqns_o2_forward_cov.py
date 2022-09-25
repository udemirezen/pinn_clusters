"""
2nd order forward equations with change of variable trick
"""
import numpy as np
import tensorflow as tf

from formulations.helpers import _data_type, to_mat_tensor
from formulations.constants import *


def B(x):
    return np.ones_like(x)


def BC_Equations(x, y, neural_net):
    """Boundary euqations for forward mode

    Note that input x & y is not used for BCs.
    They exist for consistent interface with data equations.
    """
    origin = to_mat_tensor(lx)
    u_bc = to_mat_tensor(1.0)
    h_bc = to_mat_tensor(h0)

    # v^n = du/dx boundary conditions
    v_bc = to_mat_tensor(h0 / (2 * nu_star * B(0)))

    nn_forward = neural_net(origin)
    u_pred = nn_forward[..., 0]
    v_pred = nn_forward[..., 1]
    h_pred = nn_forward[..., 2]

    return u_bc - u_pred, v_bc - v_pred, h_bc - h_pred


def forward_2nd_order_cov(x, neural_net=None):
    with tf.GradientTape(persistent=True) as tg:
        tg.watch(x)  # define the variable with respect to which you want to take derivative
        nn_forward = neural_net(x)
        u = nn_forward[..., 0:1]
        v = nn_forward[..., 1:2]  # v^n = du/dx
        h = nn_forward[..., 2:3]
        uh = h * u
        bhv = B(x) * h * v

    h_x = tg.gradient(h, x)
    u_x = tg.gradient(u, x)
    uh_x = tg.gradient(uh, x)
    bhv_x = tg.gradient(bhv, x)

    # momentum balance governing equation
    f1 = nu_star * bhv_x - h * h_x
    f2 = u_x - tf.pow(v, n)
    # mass balance governing equation
    f3 = uh_x - A0
    return f1, f2, f3
