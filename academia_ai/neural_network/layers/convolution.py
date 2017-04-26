import numpy as np


class ConvolutionLayer(object):
    '''Contains a number of filters and can convolve them with an input image
    
    It first puts the image and the filters together(forwardpropagation) and then relates pixels of a filter to the error.(backpropagation)
    
    Parameters:
    Integer padding is the amount of zeros that will be added around
    the input image. It is used to control the output dimensions of the
    convolution. It can be negative, in this case the image will be cropped
    by the given amount of pixels.
    Integer stride is used to specify the distance in pixels a filter
    is moved during convolution.
    I.e. stride=1 means every pixel gets sampled, stride=2
    means every other pixel is skipped. (But you should ignore stride because it doesn't work.)
    self.F are the filters flattened into colums.
    '''

    def __init__(self, nr_filters=3, filter_shape=(9, 9), stride=1):
        if stride != 1:
            raise NotImplementedError('stride should be =1')
        self.nr_filters = nr_filters
        self.filter_shape = filter_shape
        if(filter_shape[0] % 2 == 0 and filter_shape[1] % 2 == 0):  # not OR ?
            print('ERROR: filter shapes have to be odd numbers!')
        self.padding = (filter_shape[0] - 1) // 2
        self.stride = stride
        # Initialize random filters, Gaussian distribution with mean = 0 and
        # stdev = 1/sqrt(nr_inputs)
        filters = []
        for i in range(nr_filters):
            filters += [np.random.randn(*
                                        filter_shape) /
                        np.sqrt(np.product(filter_shape))]
        # Convert them into column format for convenient matrix multiplications
        self.filters_to_columns(filters)

    def filters_to_columns(self, filters):
        ''' Flattens filters into columns of the matrix self.F.'''

        F = [f.ravel()[:, None] for f in filters]
        self.F = np.hstack(F)

    def pprint(self):
        print(
            "Convolutional layer with",
            self.nr_filters,
            "filters of shape",
            self.filter_shape)

    def data_to_rows(self, data, shape=None, stride=1):
        ''' Flatten blocks (with given shape) in data into columns of a matrix.

        Input data should have shape (z,x,y) (with padding) and will be split
        into blocks of size shape.
        (Stride: Optional skipping of blocks; stride=1 means slide by one block,
        skip nothing;
        stride=2 means slide two blocks, skip every second block and so on.)

        in: data[z,x,y]
        out: res.reshape[z*nFilters,]
        '''

        # Find out image and filter sizes
        if shape is None:
            # Just take the last filter as representive, they should all be
            # equal
            filter_X, filter_Y = self.filter_shape
        else:
            filter_X, filter_Y = shape
        image_depth = data.shape[0]
        image_X, image_Y = data.shape[1:3]
        output_X = image_X - filter_X + 1
        output_Y = image_Y - filter_Y + 1
        # Array of indices into the first block
        start_index = np.arange(filter_X)[
            :, None] * image_Y + np.arange(filter_Y)
        # Get offset indices across the X and Y dimensions of one image (=depth
        # slice)
        offset_index = np.arange(0, output_X, stride)[
            :, None] * image_Y + np.arange(0, output_Y, stride)
        # Combine and skip stride blocks
        subimage_index = start_index.ravel()[:, None] + offset_index.ravel()
        # Repeat for all depth slices
        complete_index = np.arange(image_depth)[
            :, None, None] * (image_X) * (image_Y) + subimage_index
        # Take the calculated indices out of the data array
        out = np.take(data, complete_index)
        # and return all blocks stacked vertically
        return np.vstack(out.swapaxes(1, 2))

    def forward_prop(self, data):
        ''' In: [depth,X,Y], Out: [depth*nr_filters)][(X/stride)+1][(Y/stride)+1]
       
        Puts image and filter together.
        Uses the methods filters_to_columns and data_to_rows to convert
        the convolution into a single large matrix multiplication.
        '''
        if not (data.shape[1] % self.stride == 0):
            print('ERROR: data.shape needs to be divisible by stride!')
            # Because otherwise the backpropagation does not know how big the
            # data shape was.
        self.input_shape = data.shape
        # Pad with zeros (p > 0) or crop (p < 0) data given by the amount
        # self.padding
        p = int(self.padding)
        if p > 0:
            data_pad = np.pad(data, ((0, 0), (p, p), (p, p)),
                              mode='constant', constant_values=0)
        elif p < 0:
            data_pad = data[:, -p:p, -p:p]
        elif p == 0:
            data_pad = data
        # Determine shape of the final result
        filter_X, filter_Y = self.filter_shape
        output_shape = (
            data.shape[0] *
            self.nr_filters,
            data.shape[1] //
            self.stride,
            data.shape[2] //
            self.stride)
        # Prepare data blocks as rows in one large matrix
        D = self.data_to_rows(data_pad, stride=self.stride)
        self.i = D.copy()  # save for later use in back_prop (!)
        # Then convolution is just a matrix multiplication of the filters with
        # the data matrix
        res = np.dot(D, self.F).swapaxes(0, 1)
        # Restore the three-dimensional shape (depth, x, y) of the output data
        return np.reshape(res, output_shape, order='C')

    def back_prop(self, data, epsilon):
        ''' In: data = dEdo[z*nFilters,x/stride, y/stride], Out: dEdi[z,x,y]
        
        epsilon is the learning parameter. With epsilon you can set the learning speed.
        
        
        (Todo: dEdi shape is not generally correct now.)
        '''

        n = self.nr_filters
        dEdo = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        # First calculate dE/di = dE/do * do/di
        dEdo_split = dEdo.reshape(dEdo.shape[0] // n, n, dEdo.shape[1])
        dEdi = np.sum(np.dot(self.F, dEdo_split), axis=0)
        # sum((9*9, n)*(input-picture-amount from forward_prop, n, x*y )) =
        # sum(9*9, ipa,x*y) = (ipa, x*y)
        dEdi_shape = (data.shape[0] // n,
                      data.shape[1] * self.stride,
                      data.shape[2] * self.stride)
        # Second update the filters using data=dE/do  using  dE/dF = dEdo * i
        dEdF = np.dot(
            dEdo.reshape(
                n,
                dEdo.shape[0] //
                n *
                dEdo.shape[1]),
            self.i)
        self.F = self.F - epsilon * dEdF.swapaxes(0, 1)
        return dEdi.reshape(dEdi_shape)

    

class ConvolutionLayerSHAPE(object):
    '''Contains a number of filters and can convolve them with an input image
    
    It first puts the image and the filters together(forwardpropagation) and then relates pixels of a filter to the error.(backpropagation)
    
    Parameters:
    Integer padding is the amount of zeros that will be added around
    the input image. It is used to control the output dimensions of the
    convolution. It can be negative, in this case the image will be cropped
    by the given amount of pixels.
    Integer stride is used to specify the distance in pixels a filter
    is moved during convolution.
    I.e. stride=1 means every pixel gets sampled, stride=2
    means every other pixel is skipped. (But you should ignore stride because it doesn't work.)
    self.F are the filters flattened into colums.
    '''

    def __init__(self, in_shape, out_shape, nr_filters=3, filter_shape=(9, 9), stride=1):

        self.in_shape = in_shape
        self.out_shape = out_shape
        self.nr_filters = nr_filters
        self.filter_shape = filter_shape
        if(filter_shape[0] % 2 == 0 or filter_shape[1] % 2 == 0):
            print('ERROR: filter shapes have to be odd numbers!')
        self.padding = (filter_shape[0] - 1) // 2
        self.stride = stride
        # Initialize random filters, Gaussian distribution with mean = 0 and
        # stdev = 1/sqrt(nr_inputs)
        filters = []
        for i in range(nr_filters):
            filters += [np.random.randn(*
                                        filter_shape) /
                        np.sqrt(np.product(filter_shape))]
        # Convert them into column format for convenient matrix multiplications
        self.filters_to_columns(filters)

    def filters_to_columns(self, filters):
        ''' Flattens filters into columns of the matrix self.F.'''

        F = [f.ravel()[:, None] for f in filters]
        self.F = np.hstack(F)

    def pprint(self):
        print(
            "Convolutional layer with",
            self.nr_filters,
            "filters of shape",
            self.filter_shape,
            "in_shape",
            self.in_shape,
            "out_shape",
            self.out_shape)

    def data_to_rows(self, data, shape=None, stride=1):
        ''' Flatten blocks (with given shape) in data into columns of a matrix.

        Input data should have shape (z,x,y) (with padding) and will be split
        into blocks of size shape.
        (Stride: Optional skipping of blocks; stride=1 means slide by one block,
        skip nothing;
        stride=2 means slide two blocks, skip every second block and so on.)

        in: data[z,x,y]
        out: res.reshape[z*nFilters,]
        '''

        # Find out image and filter sizes
        if shape is None:
            # Just take the last filter as representive, they should all be
            # equal
            filter_X, filter_Y = self.filter_shape
        else:
            filter_X, filter_Y = shape
        image_depth = data.shape[0]
        image_X, image_Y = data.shape[1:3]
        output_X = image_X - filter_X + 1
        output_Y = image_Y - filter_Y + 1
        # Array of indices into the first block
        start_index = np.arange(filter_X)[:, None] * image_Y + np.arange(filter_Y)
        # Get offset indices across the X and Y dimensions of one image (=depth
        # slice)
        offset_index = np.arange(0, output_X, stride)[:, None] * image_Y + np.arange(0, output_Y, stride)
        # Combine and skip stride blocks
        subimage_index = start_index.ravel()[:, None] + offset_index.ravel()
        # Repeat for all depth slices
        complete_index = np.arange(image_depth)[:, None, None] * (image_X) * (image_Y) + subimage_index
        # Take the calculated indices out of the data array
        out = np.take(data, complete_index)
        # and return all blocks stacked vertically
        return np.vstack(out.swapaxes(1, 2))

    def forward_prop(self, data):
        ''' In: [depth,X,Y], Out: [depth*nr_filters)][(X/stride)+1][(Y/stride)+1]
       
        Puts image and filter together.
        Uses the methods filters_to_columns and data_to_rows to convert
        the convolution into a single large matrix multiplication.
        '''
        if not (data.shape[1] % self.stride == 0):
            print('ERROR: data.shape needs to be divisible by stride!')
            # Because otherwise the backpropagation does not know how big the
            # data shape was.
        self.input_shape = data.shape
        # Pad with zeros (p > 0) or crop (p < 0) data given by the amount
        # self.padding
        p = int(self.padding)
        if p > 0:
            data_pad = np.pad(data, ((0, 0), (p, p), (p, p)),
                              mode='constant', constant_values=0)
        elif p < 0:
            data_pad = data[:, -p:p, -p:p]
        elif p == 0:
            data_pad = data
        # Determine shape of the final result
        filter_X, filter_Y = self.filter_shape
        output_shape = (
            data.shape[0] * self.nr_filters,
            data.shape[1] // self.stride,
            data.shape[2] // self.stride)
        # Prepare data blocks as rows in one large matrix
        D = self.data_to_rows(data_pad, stride=self.stride)
        self.i = D.copy()  # save for later use in back_prop (!)
        # Then convolution is just a matrix multiplication of the filters with
        # the data matrix
        res = np.dot(D, self.F).swapaxes(0, 1)
        # Restore the three-dimensional shape (depth, x, y) of the output data
        return np.reshape(res, output_shape, order='C')

    def back_prop(self, data, epsilon):
        ''' In: data = dEdo[z*nFilters,x/stride, y/stride], Out: dEdi[z,x,y]
        
        epsilon is the learning parameter. With epsilon you can set the learning speed.
        
        
        (Todo: dEdi shape is not generally correct now.)
        '''

        n = self.nr_filters
        dEdo = data.reshape(data.shape[0], data.shape[1] * data.shape[2])
        # First calculate dE/di = dE/do * do/di
        dEdo_split = dEdo.reshape(dEdo.shape[0] // n, n, dEdo.shape[1])
        dEdi = np.sum(np.dot(self.F, dEdo_split), axis=0)
        # sum((9*9, n)*(input-picture-amount from forward_prop, n, x*y )) =
        # sum(9*9, ipa,x*y) = (ipa, x*y)
        dEdi_shape = (data.shape[0] // n,
                      data.shape[1] * self.stride,
                      data.shape[2] * self.stride)
        # Second update the filters using data=dE/do  using  dE/dF = dEdo * i
        dEdF = np.dot(
            dEdo.reshape(
                n,
                dEdo.shape[0] //
                n *
                dEdo.shape[1]),
            self.i)
        self.F = self.F - epsilon * dEdF.swapaxes(0, 1)
        return dEdi.reshape(dEdi_shape)

