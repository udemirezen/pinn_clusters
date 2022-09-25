import tensorflow as tf
from formulations.constants import *

# For both eqns_o1_inverse and eqns_o1_forward, the equations are actually returning the DIFFERENCE between
#the neural net prediction, and either the true data or the expectation from the equation.
def Data_Equations(x, y, neural_net):
    """The equations for matching neural net prediction and observed data"""
    u_data, h_data = y
    nn_forward = neural_net(x)
    u_pred = nn_forward[..., 0]
    h_pred = nn_forward[..., 1]
    return u_data - u_pred, h_data - h_pred

# fractional just gives a different form of the equation. Doesn't actually change the equation itself.
#exactly the same as Forward_1stOrder_Equations(fractional: bool), but with an extra "drop_mass_balance" argument.
#also, the last dimension of the model output has is +1, since we are also predicting hardness
def Inverse_1stOrder_Equations(fractional: bool):

    def inverse_1st_order(x, neural_net=None, drop_mass_balance: bool = True):
        #if drop_mass_balance=True, we literally doesn't return the mass balance equation.
        with tf.GradientTape(persistent=True) as tg:
            tg.watch(x)  # define the variable with respect to which you want to take derivative
            nn_forward = neural_net(x)
            u = nn_forward[..., 0:1]
            h = nn_forward[..., 1:2]
            B = nn_forward[..., 2:3]
            uh = h * u
        u_x = tg.gradient(u, x)
        uh_x = tg.gradient(uh, x)

        # Momentum balance governing equation
        if not fractional:
            f1 = (2 * nu_star * B) ** n * u_x - h ** n
        else:
            f1 = 2 * nu_star * B * (tf.abs(u_x) ** (1/n - 1)) * u_x - h

        if drop_mass_balance:
            return (f1, )
        else:
            # mass balance governing equation
            f2 = uh_x - A0
            return f1, f2

    return inverse_1st_order
