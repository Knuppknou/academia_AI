"""Module digits.py to handle MNIST handwritten digits sample."""


import numpy as np

# Constants - fixed properties of the dataset
image_shape = (28, 28)
image_size = (784)  # 28*28
image_shape_cut = (20, 20)


class Digit(object):
    """Represent a single handwritten digit from the MNIST dataset."""

    def __init__(
        self,
        iid=-1,
        label=10,
        array=np.zeros(
            image_size,
            dtype=np.float32)):
        self.array = np.array(array)
        self.image = array.reshape(image_shape)
        self.label = label
        self.iid = iid

    def pprint(self):
        print('Digit with ID ', self.iid, ' and solution ', self.label)

    def print_image(self):
        self.pprint()
        for i in range(image_shape[0]):
            l = ["{0:.2f}".format(self.image[i, j])[2:]
                 for j in range(image_shape[1])]
            print(' '.join(l))

            
class DigitCollection(object):
    """Store a collection of digits and provide methods for preprocessing.

    Separate set of digits into training set for learning and test set.
    Preprocess and normalize all digits given only the information from
    the training set (!) and provide method to apply this normalization
    to new instances of digits.

    Future idea: let every normalized digit update the normalization constants
    to sligthly adapt to changes in image offset, brightness, background etc.
    """

    def __init__(self, filepath='../digits/data/MNIST_academia.npz'):
        self.mean = np.zeros(image_shape_cut)
        self.var = np.ones(image_shape_cut)
        digit_list = self.load_Digits(filepath)
        training_set, test_set = self.split_list(digit_list)
        self.normalize_training_set(training_set)
        self.normalize_test_set(test_set)
        training_set = self.randomize_list(training_set)
        test_set = self.randomize_list(test_set)
        ti, ts = self.get_images_solutions(training_set)
        self.training_images = ti
        self.training_solutions = ts
        i, s = self.get_images_solutions(test_set)
        self.test_images = i
        self.test_solutions = s

    def load_Digits(self, filepath):
        """Load preprocessed MNIST_academia archive into Digit objects."""

        images, labels = self.load_images_labels(filepath)
        if images.shape[0] != labels.shape[0]:
            print('Error: Label and Image dimensions do not match!')
        dimension = images.shape[0]
        list_of_digits = []
        for i in range(dimension):
            new_Digit = Digit(i, labels[i], images[i])
            list_of_digits.append(new_Digit)
        for dig in list_of_digits:  # remove border
            dig.image = dig.image[4:-4, 4:-4]
        return list_of_digits

    def load_images_labels(self, filepath='../digits/data/MNIST_academia.npz'):
        """Return all images and all labels each as one numpy array."""

        loaded_file = np.load(filepath)
        images = np.array(loaded_file['Images'], dtype=np.float32)
        labels = np.array(loaded_file['Labels'], dtype=np.uint8)
        return images, labels

    def normalize_training_set(self, digit_list, max_factor=5):
        """Calculate mean and variance of input digits, save and normalize."""

        # find mean and variance of images
        mean = np.zeros(digit_list[-1].image.shape)
        for d in digit_list:
            mean = mean + d.image
        mean /= len(digit_list)
        var = np.zeros(mean.shape)
        for d in digit_list:
            var = var + (d.image-mean)**2
        var /= len(digit_list)
        var[var < 1/max_factor] = 1/max_factor
        # save for later use
        self.mean = mean
        self.var = var
        # apply to given set
        for d in digit_list:
            d.image = (d.image-mean) / var

    def normalize_test_set(self, digit_list):
        """Normalize input digits using saved values mean and var."""

        for d in digit_list:
            d.image = (d.image - self.mean) / self.var

    def split_list(self, digit_list, split_index=60000):
        return digit_list[:split_index], digit_list[split_index:]

    def randomize_list(self, input_list):
        return np.random.permutation(input_list)

    # Methods to conform to our cnn layout, converting labels to arrays.
    # In principle, this should not be here.
    def get_images_solutions(self, digit_list):
        """Generate list of numpy arrays with images and desired outputs."""

        images = [d.image for d in digit_list]
        solutions = [self.desired_output(d.label) for d in digit_list]
        return images, solutions

    def desired_output(self, label):
        """Note: Defining NO as -1 and YES as 1 is somewhat arbitrary
        and has to be compatible with the neural network working points!
        """

        desired = -1 * np.ones((10, 1, 1))
        desired[label, 0, 0] = 1
        return desired
