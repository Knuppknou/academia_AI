import numpy as np


class HyperbolicTangentLayer(object):
    """A layer that applies tanh element-wise.

    This layer has fixed scaling parameters for the tanh and
    does not adjust weights during training.
    """

    def __init__(self, m=2/3, alpha=1.7159, beta=0):
        """Create new HyperbolicTangentLayer.

        Can specify scaling parameters m, alpha and beta:
        output = alpha * tanh(m * input) + beta * input.
        """
        
        self.m = m
        self.alpha = alpha
        self.beta = beta

    def forward_prop(self, data):
        self.newdata = self.alpha * np.tanh(self.m * data)
        return self.newdata + self.beta * data

    def back_prop(self, data):
        return data * (self.m * self.alpha -
                       (self.m * self.newdata**2) + self.beta)

    def pprint(self):
        print(
            "tangent hyperbolicus layer with m=",
            self.m,
            "and alpha=",
            self.alpha,
            "and beta=",
            self.beta)
