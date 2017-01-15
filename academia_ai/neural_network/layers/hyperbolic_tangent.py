import numpy as np


class HyperbolicTangentLayer(object):
    """INSERT DOCUMENTATION HERE!"""

    def __init__(self, m=2 / 3, alpha=1.7159, beta=0):
        self.m = m
        self.alpha = alpha
        self.beta = beta

    def forward_prop(self, data, debug=False):
        self.newdata = self.alpha * np.tanh(self.m * data)
        return self.newdata + self.beta * data

    def back_prop(self, data, debug=False):
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
