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

    def back_prop(self, data, learning_rate=0):
        return (self.i >= 0) * data

    
class ReLuLayerSHAPE(object):
    """Applies ouput=max(input, 0) element-wise.

    ReLu: Rectified linear unit. Introduces nonlinearity to the network.
    """

    def __init__(self, in_shape):
        self.in_shape = in_shape
        self.out_shape = out_shape

    def pprint(self):
        print("ReLu Layer with in_shape = out_shape =",
             self.out_shape)

    def forward_prop(self, data):
        self.i = (data >= 0) * data  # save for back_prop
        return (data >= 0) * data

    def back_prop(self, data, learning_rate=0):
        return (self.i >= 0) * data