import numpy as np
import tensorflow as tf
from tensorflow.python.types.core import Value

from formulations.helpers import _data_type, to_mat_tensor
from formulations.constants import *
import math as m

#sum of two gaussians subtracted from 1
def gaussian(x,mu1,sig1,mu2,sig2):
  return 1-0.85*(tf.math.exp((-0.5*(x-mu1)**2)/(sig1**2))+tf.math.exp((-0.5*(x-mu2)**2)/(sig2**2)))

def B(x):
    # Sets the shape of our hardness function
    
    #CONSTANT B
   # return np.ones_like(x)

    #SINUSOIDAL B 
    #pi_value = tf.constant(m.pi, dtype=_data_type)
    #return 2*tf.math.cos((5*pi_value)*x)+1
    #return tf.math.cos((5*pi_value)*x)+2

    #DOUBLE GAUSSIAN
    return gaussian(x, 0.3, 0.03, 0.7, 0.03)


def BC_Equations(x, y, neural_net):
    """Boundary equations for forward mode

    Note that input x & y is not used for BCs.
    They exist for consistent interface with data equations in inverse mode.
    """
    origin = to_mat_tensor(lx)
    u_bc = to_mat_tensor(1.0)
    h_bc = to_mat_tensor(h0)

    nn_forward = neural_net(origin)
    u_pred = nn_forward[..., 0]
    h_pred = nn_forward[..., 1]
    return u_bc - u_pred, h_bc - h_pred


def Forward_1stOrder_Equations(fractional: bool):

    def forward_1st_order(x, neural_net=None):
        with tf.GradientTape(persistent=True) as tg:
            tg.watch(x)  # define the variable with respect to which you want to take derivative
            nn_forward = neural_net(x)
            u = nn_forward[..., 0:1]
            h = nn_forward[..., 1:2]
            uh = h * u
        u_x = tg.gradient(u, x)
        uh_x = tg.gradient(uh, x)

        # Momentum balance governing equation
        if not fractional:
            f1 = (2 * nu_star * B(x)) ** n * u_x - h ** n
        else:
            f1 = 2 * nu_star * B(x) * (tf.abs(u_x) ** (1/n - 1)) * u_x - h
        # Mass balance governing equation
        f2 = uh_x - A0
        return f1, f2

    return forward_1st_order
