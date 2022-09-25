from typing import Dict

import tensorflow as tf


class SquareLoss:
    """Calculate square loss from given physics-informed equations and data equations

    Note that Data equation can be used to set boundary conditions too.
    """

    def __init__(self, equations, equations_data, gamma: float) -> None:
        """
        Args:
            equations:
                an iterable of callables, with signatrue function(x, neural_net)
            equations_data:
                an iterable of callables, with signatrue function(x, y, neural_net)
            gamma:
                the normalized weighting factor for equation loss and data loss,
                loss = gamma * equation-loss + (1 - gamma) * data-loss
        """
        self.eqns = equations
        self.eqns_data = equations_data
        self.gamma = gamma

    def __call__(self, x_eqn, data_pts, net) -> Dict[str, tf.Tensor]:
        equations = self.eqns(x=x_eqn, neural_net=net)
        #equations = self.eqns(x=x_eqn, neural_net=net, drop_mass_balance = False)
        x_data, y_data = data_pts
        datas = self.eqns_data(x=x_data, y=y_data, neural_net=net)
        loss_e = sum(tf.reduce_mean(tf.square(eqn)) for eqn in equations)
        loss_d = sum(tf.reduce_mean(tf.square(data)) for data in datas)
        loss = (1 - self.gamma) * loss_d + self.gamma * loss_e
        return {"loss": loss, "loss_equation": loss_e, "loss_data": loss_d}
