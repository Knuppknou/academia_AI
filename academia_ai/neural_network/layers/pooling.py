import numpy as np


class PoolingLayer(object):
    """Reduce the size of input images by pooling pixels together.

    The typus can be
    "mean": ...
    "max": 
    Parameter factor specifies...?
    """

    def __init__(self, factor=5, typus='mean'):
        self.typus = typus  # can be 'max' 'mean' ('min' 'absmax')
        # factor=5 means that the matrix will be compressed 1:4. (for size 20)
        self.factor = factor

    # Remark: propagations can be done seperately of the depth
    # in: data[depth][size][size]
    # out: data[depth][size/self.factor][size/self.factor]
    def forward_prop(self, data, debug=False):
        newdata = np.zeros(
            (data.shape[0],
             (data.shape[1]) //
                self.factor,
                (data.shape[2]) //
                self.factor))

        newsize = (int)((data[0].shape[0]) / self.factor)
        newimage = np.zeros((newsize, newsize))
        self.dodi = np.zeros(data.shape)  # for backprop typus max
        for depth in range(data.shape[0]):
            for x in range(newsize):
                for y in range(newsize):
                    submatrix = data[depth][
                        x *
                        self.factor:(
                            x +
                            1) *
                        self.factor,
                        y *
                        self.factor:(
                            y +
                            1) *
                        self.factor]
                    if self.typus == 'max':
                        # TODO gaus distribution
                        # find index of the greatest value
                        ind = np.add(
                            np.unravel_index(
                                np.argmax(submatrix),
                                submatrix.shape),
                            (x * self.factor,
                             y * self.factor))
                        # if the pixel had an influence then it is now 1, else
                        # 0
                        self.dodi[depth][ind[0]][ind[1]] = 1
                        newimage[x, y] = submatrix.max()
                    elif self.typus == 'mean':
                        newimage[x, y] = np.mean(submatrix) * \
                            np.sqrt(data.shape[1] * data.shape[2] / (self.factor**2))

            newdata[depth] = newimage
        if self.typus == 'mean':
            self.dodi = (1 / self.factor**2) * \
                np.sqrt(data.shape[1] * data.shape[2] / (self.factor**2))
        if debug:
            print("Calculated forward propagation for pooling layer")
        # print(self.backdata)
        return newdata

    # in: data[depth][size/self.factor][size/self.factor]
    # out: data[depth][size][size]
    def back_prop(self, data, epsilon, debug=False):
        # no internal weight-derivation needed in the pooling-layer
        # have: dE/do[d,i,j]=data[d,i,j]
        # want: dE/di[d,i',j']=dF/do[d,i,j]*do[d,i,j]/di[d,i',j']
        if self.typus == 'max':
            newdata = np.zeros(
                (data.shape[0],
                 (data.shape[1]) *
                    self.factor,
                    (data.shape[2]) *
                    self.factor))
            for d in range(newdata.shape[0]):
                for i in range(newdata.shape[1]):
                    for j in range(newdata.shape[2]):
                        # TODO: we might have to make sure this makes sense,
                        # when new typuses are added.
                        newdata[d, i, j] = self.dodi[d, i, j] * \
                            data[d, (i) // self.factor, (j) // self.factor]
        elif self.typus == 'mean':
            newdata = np.repeat(data, self.factor, axis=1)
            newdata = np.repeat(newdata, self.factor, axis=2)
            newdata *= self.dodi
        if debug:
            print("Calculated backpropagation for pooling layer")
        return newdata

    def pprint(self):
        print(
            "Pooling layer with compression factor",
            self.factor)
