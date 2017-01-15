import numpy as np


class FullyConnectedLayer(object):
    """A layer with weights connecting every input to every output node."""

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape
        # Initialize random weight matrix, Gaussian with mean = 0
        # and stdev = (input_size)^(-1/2)
        self.W = np.random.randn(np.product(input_shape), np.product(
            output_shape)) / np.sqrt(np.product(input_shape))

    def pprint(self):
        print('FullyConnectedLayer with input_shape', self.input_shape,
              'and output_shape', self.output_shape)
        W = self.W
        print('Properites of weights:',
              'MIN=', round(np.min(W), 2), 'MAX=', round(np.max(W), 2),
              'MEAN=', round(np.mean(W), 2), 'VAR=', round(np.var(W), 2))

    def forward_prop(self, data):
        """Propagate data forward by multiplying data with weight matrix W.

        Additionally save the input data in flattened form for reuse
        in back_prop. Shape of data must be input_shape; return numpy
        array with result of shape output_shape.
        """

        # Save flat input for backprop later
        self.i = data.ravel().copy()
        # Calculate output, the matrix product of flattened input and weights
        o = np.dot(self.i, self.W)
        # and reshape to output_shape
        return o.reshape(self.output_shape)

    def back_prop(self, derror_dout, learning_rate):
        """Propage error derivative back and adjust weights W.

        Short variable  names:
        d: derivative
        E: error
        o: output of this layer
        i: input to this layer
        """

        # Flatten error derivative
        dEdo = derror_dout.ravel()
        # First calculate dE/di = dE/do * do/di
        dEdi = np.dot(dEdo, self.W.T)
        # Secondly, update weights with dE/dW = dE/do * do/dW
        # scaled with the learning_rate
        dEdW = self.i[:, None] * dEdo
        self.W = self.W - learning_rate * dEdW
        # Return dE/di for the next back_prop step in the previous layer
        return dEdi.reshape(self.input_shape)
