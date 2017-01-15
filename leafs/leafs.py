import numpy as np
import matplotlib.pyplot as plt
# from PIL import Image
# import os
import pickle


class Leaf(object):
    '''  '''

    def __init__(self, iid=-1, label=-1, matrix=np.zeros((1, 1)), labelstr=''):
        self.image = matrix
        if self.image.shape == (1, 1):
            print('Error: no input matrix given')
        self.label = label
        self.labelstr = labelstr
        self.iid = iid

    def pprint(self):
        print('Leaf with ID ', self.iid, ' and solution ', self.label)
        
    def plot_colours(matriz):
        '''Plots all three diffrent colours of a leaf.

        Input: matrix (3D)
        '''
        plt.subplot(2, 3, 1)
        plt.imshow(matriz[:, :, 0], cmap=plt.cm.Reds)
        # plt.clim(0,256)
        # plt.colorbar()

        plt.subplot(2, 3, 2)
        plt.imshow(matriz[:, :, 1], cmap=plt.cm.Greens)
        # plt.clim(0,256)
        # plt.colorbar()

        plt.subplot(2, 3, 3)
        plt.imshow(matriz[:, :, 2], cmap=plt.cm.Blues)
        # plt.clim(0,256)
        # plt.colorbar()

        plt.tight_layout()
        plt.show()

#############################################


def save_Leafs(Leafs, path="Leafs.pkl"):
    f = open(path, 'wb')
    pickle.dump(Leafs, f)
    f.close()
    print('Saved "Leafs" with', len(Leafs), 'leafs in', path, '.')


def load_Leafs(path="Leafs.pkl"):
    f = open(path, "rb")
    Leafs = pickle.load(open(path, "rb"))
    f.close()
    print('Loades "Leafs" from path', path, ',with', len(Leafs), 'Leafs.')
    return Leafs


def plot_colours(matriz):
    '''Plots all three diffrent colours of a leaf.

    Input: matrix (3D)
    '''
    plt.subplot(2, 3, 1)
    plt.imshow(matriz[:, :, 0], cmap=plt.cm.Reds)
    # plt.clim(0,256)
    # plt.colourbar()

    plt.subplot(2, 3, 2)
    plt.imshow(matriz[:, :, 1], cmap=plt.cm.Greens)
    # plt.clim(0,256)

    plt.subplot(2, 3, 3)
    plt.imshow(matriz[:, :, 2], cmap=plt.cm.Blues)
    # plt.clim(0,256)

    plt.tight_layout()
    plt.show()
