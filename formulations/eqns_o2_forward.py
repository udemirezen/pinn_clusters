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

    # dudx boundary conditions
    dudx_bc= (h0 / (2 * nu_star * B(0))) ** n
    dudx_bc = to_mat_tensor(dudx_bc)

    with tf.GradientTape() as tg:
        tg.watch(origin)
        nn_forward = neural_net(origin)
        u_pred = nn_forward[..., 0]
        h_pred = nn_forward[..., 1]
    dudx_pred = tg.gradient(u_pred, origin)

    return u_bc - u_pred, h_bc - h_pred, dudx_bc - dudx_pred


def forward_2nd_order(x, neural_net=None):
    with tf.GradientTape(persistent=True) as tg2:
        tg2.watch(x)
        with tf.GradientTape(persistent=True) as tg1:
            tg1.watch(x)  # define the variable with respect to which you want to take derivative
            nn_forward = neural_net(x)
            u = nn_forward[..., 0:1]
            h = nn_forward[..., 1:2]
            uh = h * u
        h_x = tg1.gradient(h, x)
        u_x = tg1.gradient(u, x)
        uh_x = tg1.gradient(uh, x)
        mom_lhs = nu_star * B(x) * u_x * (tf.sqrt(u_x ** 2) ** (1/n - 1))
    mom_lhs_x = tg2.gradient(mom_lhs, x)

    # momentum balance governing equation
    f1 = mom_lhs_x - h * h_x
    # mass balance governing equation
    f2 = uh_x - A0
    return f1, f2
