"""Contains basic neural network structure that will host neural layers."""

import os.path
import numpy as np
import pickle


class ConvolutionalNeuralNet(object):
    """Convolutional neural network holds layers and methods to train and test.

    It is built from  several neural layers, as for example convolutional
    layers, pooling layers or fully connected layers.

    Usage:
    1. Create the layers that will be used in the network and create an empty
       ConvolutionalNeuralNet.
    2. Add all layers using the add_layer() method in the order they should be
       placed in the net.
    3. Use the network's forward_prop() to iterate a given input array through
       the entire network.
    4. Do back_prop() and check if the network improved.
    5. Repeat 3-4 as often as one can afford.

    Example:
    -> See ipython notebooks for demonstrations.


    Notes / To Do:
    - One should alternate forward_prop and back_prop, because each call to
      forward_prop saves data necessary for back_prop.
      (...Should handle this for the user)
    - Should include some mechanism to avoid overfitting.
    - Should keep track of all input and output dimensions. Should build the
      right dimensions for the user from simple arguments. It should be
      impossible to connect layers that have incompatible dimensions.
    - Tuning the learning_rate is crucial. Should provide functions to help.
    - The save and load using pickling is pretty bad, as changes to the source
      code quickly render saved files useless.
 
    Implementation details:
    Each layer must provide methods forward_prop and back_prop to move
    through the network layers.

    Each single layer forward_prop method takes data as input in the form of a
    three-dimensional numpy array data[depth][x][y] and return the propagated
    data which is also three-dimensional but all individual dimensions might
    change. The last layer of the network consist of output classes and during
    training, the output of the last layer is compared to the solution using
    an error function.

    Backpropagation of each single layer, back_prop, consists of two steps:
    1. From the input dError/dOutput calculate dError/dInput to pass it to
       the previous layer in backpropagation.
    2. In case of variable weights, calculate dError/dWeights to adjust
       the weights in the direction of decreasing error.

    """

    def __init__(self):
        """ Initialize emtpy convolutional neural net, ready to add layers."""
        
        self.layers = []
        self.success_list = []  #  added recently--> problem with older nets
        
    def add_layer(self, input_layer):
        """Append input_layer to the list of layers in this net."""

        # Layers must provide froward_prop and back_prop
        check_forward = getattr(input_layer, 'forward_prop', None)
        if not callable(check_forward):
            print('Error: Provided layer does not have forward_prop! ')
            return None
        check_backward = getattr(input_layer, 'back_prop', None)
        if not callable(check_backward):
            print('Error: Provided layer does not have back_prop! ')
            return None
        # Add the layer at the end of the networks list of layers
        self.layers.append(input_layer)

    def pprint(self):
        """Print network layers to standard output."""

        print('Convolutional Neural Network containing the following',
              len(self.layers), "layers:")
        for l in self.layers:
            l.pprint()  # just assumes this method exists...

    def forward_prop(self, array, desired_output=None,
                     save_intermediate=False):
        """ Propagate the input data forward through the entire network.

        Start with the first layer in the net and iteratively call all
        forward_prop methods of the individual layers.
        Take numpy array as input with 2 or 3 axes, usually an image with
        two dimensions [x][y] or three dimensions [z][x][y].
        The result after iterating through all layers is returned.
        If the flag save_intermediate is True, the input and output of
        every layer is saved into the list self.intermediate_results.
        """

        # To have three-dimensional arrays throughout for consistency
        if len(array.shape) == 2:
            array = np.array([array])

        # Propagate input array through all the network layers
        if save_intermediate:
            self.intermediate_results = [array]
            for layer in self.layers:
                array = layer.forward_prop(array)
                self.intermediate_results.append(array)
        else:
            for layer in self.layers:
                array = layer.forward_prop(array)
        # Calculate the deviation from desired_output, if given
        # if desired_output is not None:
        #     self.error_function(array, desired_output)
        return array

    def error_function(self, acutal_output, desired_output):
        """Calculate error and derivative w.r.t. output.

        Inputs as numpy arrays of the same shape.
        Error function: 1/2 * (acutal-desired)**2
        """

        derror_dout = acutal_output - desired_output
        error = np.sum(derror_dout * derror_dout) / 2
        return error, derror_dout

    """ OLD CODE
     self.dEdout = np.zeros((desired_output.shape[0],1,1))
        for i in range(deviation.shape[0]):
            self.dEdout[i][0][0]=deviation[i]
    """

    def back_prop(self, derror_dout, learning_rate=0,
                  save_intermediate=False):
        """Propagate backwards through all network layers and adjust weights.

        Start with the derivative of the error function in the last layer
        and iteratively call all back_prop methods to the first layer.
        Adjust the network weights if learning_rate is set larger than zero.
        If flag save_intermediate is True, save intermediate_results from
        backpropagation in list starting with the input derror_dout.
        """

        if save_intermediate:
            self.intermediate_results = [derror_dout]
            for layer in self.layers[::-1]:
                derror_dout = layer.back_prop(derror_dout, learning_rate)
                self.intermediate_results.append(derror_dout)
        else:
            for layer in self.layers[::-1]:
                derror_dout = layer.back_prop(derror_dout, learning_rate)
                
    def train(self, training_list, solution_list, learning_rate, iterations=1):
        """Iterate through training_list iterations times and adjust weights.

        As input, provide a list of solutions corresponding to the training_list
        and provide a fixed learning rate.
        """
        
        for i in range(iterations):
            for t, s in zip(training_list, solution_list):
                o = self.forward_prop(t)
                if o.shape != s.shape:
                    print('Warning: Shapes of forward_prop output and',
                          'solution_list entry do not match! Please check',
                          'shape of solution_list.')
                e, dedo = self.error_function(o, s)
                self.back_prop(dedo, learning_rate, False)

    def train2(self, training_images, training_solutions, test_images,
               test_solutions, learning_rate, iterations=1,
               printTimeConsumtion = False):
        """Works simillary as train but tracks learning sucess."""

        # if list is empty add the untrained status to sucess_point
        if len(self.success_list) == 0:
            self.success_list.append(self.test_net(test_images, test_solutions))
        if printTimeConsumtion:
            from timeit import default_timer as timer  # maybe valuable here!

        for i in range(iterations):
            if printTimeConsumtion:
                start = timer()
                
            for t, s in zip(training_images, training_solutions):
                o = self.forward_prop(t)
                if o.shape != s.shape:
                    print('Warning: Shapes of forward_prop output and',
                          'solution_list entry do not match! Please check',
                          'shape of solution_list.')
                e, dedo = self.error_function(o, s)
                self.back_prop(dedo, learning_rate, False)
            
            if printTimeConsumtion:
                end = timer()
                print('Consumed time for ', i, ' iteration:', end-start, ' seconds.')
            self.success_list.append(self.test_net(test_images, test_solutions))

    def plott_lernSuccess(self, plt):
        """Plots the success of the network on y-axis and the corresponding
           training iteration on x-axis
        """
        
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.success_list)
        plt.ylim(0, 1)
        plt.xlabel('Lernschritt')
        plt.ylabel('Lernerfolg in Prozent')
        plt.title('Lernfortschritt')
        # plt.yticks(np.linspace(0, 1, 100), labels)
        plt.tight_layout()
        plt.grid(True)

    def test_net(self, test_sample, solutions):
        """Given list of samples with solutions (categories), benchmarks net.

        Inputs should be list or arrays stacked in first dimension with
        the same length.
        Returns a number in [0,1] corresponding to the probability of
        successful classification of the test_sample.
        """

        success = 0
        if len(test_sample) == 0:
            return success
        for sample, solution in zip(test_sample, solutions):
            if np.argmax(self.forward_prop(sample)) == np.argmax(solution):
                success += 1
        return success / len(test_sample)

    def save(self, path):
        """Save (pickle) the network to disk at given path."""
        filename, ending = os.path.splitext(path)
        if ending != '.pkl':
            path = path + '.pkl'
        f = open(path, 'wb')
        pickle.dump(self, f)
        f.close()
        print('Saved net in file:', path)

    @classmethod
    def load(cls, path):
        """Load and return a saved (pickled) network from given path."""
        file = open(path, 'rb')
        net = pickle.load(file)
        file.close()
        return net
