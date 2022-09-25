import tensorflow as tf

from formulations.constants import *


def Data_Equations(x, y, neural_net):
    """The equations for matching neural net prediction and observed data"""
    u_data, v_data, h_data = y
    nn_forward = neural_net(x)
    u_pred = nn_forward[..., 0]
    v_pred = nn_forward[..., 1]
    h_pred = nn_forward[..., 2]
    b_pred = neural_net(x[0:1, :]) # bc for B
    return u_data - u_pred, v_data - v_pred, h_data - h_pred, b_pred - 1


def inverse_2nd_order(x, neural_net=None, drop_mass_balance: bool = True):
    """2nd order inverse with change of variable trick"""
    with tf.GradientTape(persistent=True) as tg:
        tg.watch(x)  # define the variable with respect to which you want to take derivative
        nn_forward = neural_net(x)
        u = nn_forward[..., 0:1]
        v = nn_forward[..., 1:2]  # v^n = du/dx
        h = nn_forward[..., 2:3]
        B = nn_forward[..., 3:4]
        uh = h * u
        bhv = B * h * v

    u_x = tg.gradient(u, x)
    h_x = tg.gradient(h, x)
    uh_x = tg.gradient(uh, x)
    bhv_x = tg.gradient(bhv, x)

    # Momentum balance governing equation
    f1 = nu_star * bhv_x - h * h_x
    f2 = u_x - tf.pow(v, n)

    if drop_mass_balance:
        return f1, f2
    else:
        # mass balance governing equation
        f3 = uh_x - A0
        return f1, f2, f3
