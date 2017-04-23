import numpy as np


class SigmoidLayer(object):
    ''' With the sigmoid function the data pixels are set on 0 or 1 generally.
    
    tp is the turning point of the sigmoid function.
    m is the slope of the function.
    Todo: implement Gauss distribution
    '''
    class Sigmoid(object):

        def sigmoid(x, tp, m):
            y = 1 / (1 + (np.exp(-m * (x - tp))))
            return y

        sigmoid_vector = np.vectorize(sigmoid)

        def __init__(self, tp=0, m=20):
            self.tp = tp
            self.m = m

        # ignore self, data[depth][i][j]
        def forward_prop(self, data):
            new_data = np.zeros(data.shape)
            new_data = self.sigmoid_vector(data, self.tp, self.m)
            return new_data

        def back_prop(self, data):
            return self.m * self.sigmoid_vector(data, self.tp, self.m) * (
                1 - self.sigmoid_vector(data, self.tp, self.m))

    def __init__(self, tp=0.5, m=20, shape=(2, 2), learning=True):
        self.shape = shape
        self.learning = learning

        sl_list = []
        tp_list = []
        m_list = []
        # produce a vector with the Sigmoid(data) of each olddata
        #                 with the turning_point of each
        #                 with the slope m of each
        for x in range(np.product(self.shape)):
            sl_list.append(SigmoidLayer.Sigmoid(tp, m))
            tp_list.append(tp)
            m_list.append(m)
        self.sl_vector = np.array(sl_list)
        self.tp_vector = np.array(tp_list)
        self.m_vector = np.array(m_list)

    def pprint(self):
        print("sigmoid layer with shape=",  self.shape)

    def forward_prop(self, data, debug=False):
        if not(data.shape == self.shape):
            print("Error: shapes do not match")
        self.i_vector = data.ravel().copy()
        o_list = []
        for x in range(np.product(self.shape)):
            o_list.append(self.sl_vector[x].forward_prop(self.i_vector[x]))
        self.o_vector = np.array(o_list)
        return self.o_vector.reshape(self.shape)

    def back_prop(self, data, epsilon=0):
        data = data.ravel().copy()
        o = self.o_vector
        dedtp = data * (self.m_vector * o * (o - 1))
        # print("dedtp=",dedtp)
        dedm = data * ((self.tp_vector - self.i_vector) * o * (o - 1))
        # print("dedm=",dedm)

        if (self.learning or epsilon != 0):
            # newtp and newm produced with the learningrate epsilon and tp and
            # m updated
            self.tp_vector = self.tp_vector - epsilon * dedtp
            # TODO: Why is the learning of m so much slower?
            self.m_vector = self.m_vector - 100 * epsilon * dedm
            for i in range(len(self.sl_vector)):
                self.sl_vector[i].m = self.m_vector[i]
                self.sl_vector[i].tp = self.tp_vector[i]
        # dedi
        dEdi = data * (self.m_vector * o * (1 - o))
        return dEdi.reshape(self.shape)
