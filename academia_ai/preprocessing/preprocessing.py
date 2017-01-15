import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def preprocessing(path, cfactor = 50, square_side = 2000, debug=False):
    '''Preprocesses a picture

    Input: path, compressionfactor = 50, squareside of new image= 2000, debug=False
    Output: matrix that can be use as a CNN Input
    '''

    im = center_leaf(path, square_side, debug)
    new_shape = im.size[0] // cfactor
    new_im = im.resize((new_shape, new_shape))  # makes the resolution smaller

    matriz = np.array(new_im)  # convert image to numpy matrix
    matriz ^= 0xFF  # invert matrix
    oneD_matriz = matriz[:, :, 1]  # only looking at one dimension, 1 = green

    if debug:
        print(
            'Image “',
            path,
            '“ opened with size:',
            im.size,
            'and mode:',
            im.mode)
        print('compressed the square-image with lenght :',
              oneD_matriz.shape[0], ' with factor:', cfactor)
        print('output matrix has shape:', oneD_matriz.shape)
        plt.imshow(oneD_matriz)
        plt.tight_layout()
        plt.show()

    return oneD_matriz


def center_leaf(path, square_side=2000, debug=False):
    '''
    region we look at, because of the border we found with overlapping
    centers a square on the leaf

    input: path of image
           square_side of matriz thats cut away
           debug

    output: cut image
    '''
    up = 500
    down = 2900
    left = 300
    right = 4000

    im = Image.open(path).convert('RGB')
    matriz = np.array(im)  # convert image to numpy matrix
    matriz ^= 0xFF  # invert matrix
    oneD_matriz = matriz[:, :, 1]  # only looking at one dimension, 1 = green

    if debug:
        plt.subplot(1, 3, 1)
        plt.imshow(oneD_matriz)

        plt.subplot(1, 3, 2)
        plt.imshow(oneD_matriz[up:down, left:right])

    # count the total weight to normalize the sum of all weights to one
    total_weight_x = 0
    total_weight_y = 0
    for x in range(up, down, 5):  # 5 for step
        for y in range(left, right, 5):
            # 280 is treshhold, all below becomes 0, above 1
            total_weight_x += (oneD_matriz[x, y] >= 180)
            total_weight_y += (oneD_matriz[x, y] >= 180)

    # calculate the medium point to centralize the square around the medium
    # pixel (the square is the input of the CNN)
    meanx = 0
    meany = 0
    for x in range(
            up,
            down,
            5):  # only take evry 5 field for speed reasons -->makes code 25x faster
        for y in range(left, right, 5):
            # count all cordinates together but weight them differently,devide
            # by total_weight
            meanx += (oneD_matriz[x, y] >= 180) * x / total_weight_x
            meany += (oneD_matriz[x, y] >= 180) * y / total_weight_y

    # select new area of the matrix, that is the input for CNN
    s = square_side // 2
    box = (meany - s, meanx - s, meany + s, meanx + s)
    new_image = im.crop(box)  # crop is Pill function
    im.close()

    if debug:
        print(
            'meanx and meany are:',
            meanx,
            meany,
            ' the square is set with side',
            square_side)

        # plot a graphic of the way this function works
        oneD_matriz = oneD_matriz[up:down, left:right]
        for x in range(0, oneD_matriz.shape[0]):
            for y in range(0, oneD_matriz.shape[1]):
                oneD_matriz[x, y] = (oneD_matriz[x, y] >= 180) * 1
        plt.subplot(1, 3, 3)
        plt.imshow(oneD_matriz, cmap='gray')

    # return new_image
    return new_image


def create_Leafs(root_path):
    # TODO: find out the cutting size, find out contraction factor
    # default might be wrong

    #root_path ='/Users/Dino/Desktop/Data/'
    Leafs = []
    iid = 0
    treeid = 0
    print(root_path)
    for root, dirs, files in os.walk(root_path, topdown=True):
        for d in dirs:
            for root, dirs, files in os.walk(os.path.join(root_path, d)):
                for f in files:
                    if f[0] == 'I':  # To make sure, that we don't open .DStore

                        matriz = preprocessing(os.path.join(root_path, d, f))
                        new_leaf = leafs.Leaf(iid, treeid, matriz[:, :], d)
                        Leafs += [new_leaf]
                        iid += 1
            treeid += 1
    return Leafs


def find_overlap(root_path):
    #root_path ='/Users/Dino/Desktop/Data/'
    maximum = np.zeros((3456, 4608))
    for root, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            im_path = (os.path.join(root, name))
            if name[
                    0] == 'I':  # cheatyh way of making sure its an image, because there were some other files in the directory
                image = Image.open(im_path)
                matriz = preprocessing.image_convert(image.convert('RGB'))
                maximum = np.maximum(maximum, matriz[:, :, 0])
                maximum = np.maximum(maximum, matriz[:, :, 1])
                maximum = np.maximum(maximum, matriz[:, :, 2])
                image.close()
    return maximum