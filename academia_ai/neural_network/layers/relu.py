class ReLuLayer(object):
    """Applies ouput=max(input, 0) element-wise.

    ReLu: Rectified linear unit. Introduces nonlinearity to the network.
    """

    def __init__(self):
        pass

    def pprint(self):
        print("ReLu Layer")

    def forward_prop(self, data):
        self.i = (data >= 0) * data  # save for back_prop
        return (data >= 0) * data

    def back_prop(self, data):
        return (self.i >= 0) * data
