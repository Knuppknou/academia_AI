"""Artificial Intelligence python package of the academia Wattwil.

Created during the first project 2016/2017, which focussed on neural
networks. A convolutional neural network was prgrammed and used to 
identify handwritten digits and photographs of tree leafs.

Documentation mostly inside the modules; tutorials in the folder examples.
"""


"""The following statements expose the commonly used modules and
classes to the user, hiding the internal folder structure."""
# neural network code
#from . import neural_network
from .neural_network.cnn import ConvolutionalNeuralNet
# individual neural network layers
from .neural_network.layers.convolution import ConvolutionLayer
from .neural_network.layers.fully_connected import FullyConnectedLayer
from .neural_network.layers.hyperbolic_tangent import HyperbolicTangentLayer
from .neural_network.layers.pooling import PoolingLayer
from .neural_network.layers.relu import ReLuLayer
from .neural_network.layers.sigmoid import SigmoidLayer
# digits - code specific to mnist handwritten digit problem
from .digits import digits
# leafs - code specific to leaf classification
from .leafs import leafs
import academia_ai.preprocessing
import academia_ai.plotting
# flake8: noqa