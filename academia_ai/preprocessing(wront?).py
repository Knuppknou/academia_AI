import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from .leafs import leafs

def normalize(dataset):
    '''normalize data so that over all imeges the pixels on place (x/y) have mean = 0 and are standart distributed'''
    # calculate the mean
    mean=np.zeros(dataset[0].image.shape)
    for lea in dataset:
        mean=mean+lea.image
    mean/=len(dataset)
    
    #calculating the variance
    var=np.zeros(dataset[0].image.shape)
    for lea in dataset:
        var=var+(lea.image-mean)**2
    var/=len(dataset)
    f=0.1
    var=(var-f>=0)*(var-f)+f  # caps the minimal 
    for lea in dataset:
        lea.image=(lea.image-mean)/var
    
def createTrainingAndTestingList(directory, shuffle = True):
    l_train = [] # training set
    l_test = [] # testing set

    for n in range (7):
        matrices = np.load(os.path.join(directory, str(n)+'.npy'))
        
        for i in range(799): # 2x800 for training
            l_train += [leafs.Leaf(i+n*1000, n, matrices[i]/255)]
            # Leaf(iid,label,matrix,labelstr) *labelstr deleted ..too difficult

        for i in range(799,899): # 2x100 for testing
            l_test += [leafs.Leaf(i+n*1000, n, matrices[i]/255)]
    if shuffle: 
        np.random.shuffle(l_train)
        np.random.shuffle(l_test)
    return([l_train,l_test])
    
        
def collectData(root_path, save_path, cfactor, overwrite = False):
    '''processes images from root_path one-by-one and save them in same directory
    collect them try by tre, set their labels and return a training and a testing list'''
    sizeOfMatrixes = int(2000//cfactor)

    #processing images to arrays one-by-one and save inplace
    iid = 0
    for (root, dirnames, filenames) in os.walk(root_path, topdown = True):
        for f in filenames:
            if f.endswith('.JPG'):
                savepath = os.path.join(root, os.path.splitext(f)[0])
                savepath += ('_' + str(sizeOfMatrixes) + 'x' + str(sizeOfMatrixes)) # for example + _50x50
                if(not(os.path.isfile(savepath+'.npy')) or overwrite):
                    matriX = centr_cut_compress(os.path.join(root, f), cfactor)
                    np.save(savepath, matriX, allow_pickle=False)
                iid += 1
    
    # collecting all arrays from tree i into one big folder calld i.npy            
    for i in range (0,8):
        tree_path = os.path.join(root_path, str(i))
        tree_save_path = os.path.join(save_path, str(sizeOfMatrixes) + 'x' + str(sizeOfMatrixes) ,str(i))
        leaf_list = []
        for (root, dirnames, filenames) in os.walk(tree_path , topdown=True):  
            for f in filenames:
                if f.endswith('_' + str(sizeOfMatrixes) + 'x' + str(sizeOfMatrixes) + '.npy'):
                    leaf_list.append(np.load(os.path.join(root, f)))
        leaf_array = np.array(leaf_list)
        np.save(tree_save_path, leaf_array, allow_pickle=False)

def desired_output(label):
    res = -1 * np.ones((7,1,1))
    res[label, 0, 0] = +1
    return res

def centr_cut_compress(path, cfactor = 50, square_side = 2000, debug=False):
    '''centers, cuts and compresses a picture 

    Input: path, compressionfactor = 50, squareside of new image= 2000, debug=False
    Output: matrix that can be use as a CNN Input
    '''

    im = center_leaf(path, square_side)
    new_shape = im.size[0] // cfactor
    new_im = im.resize((new_shape, new_shape))  # makes the resolution smaller

    matriz = np.array(new_im)  # convert image to numpy matrix
    matriz ^= 0xFF  # invert matrix
    oneD_matriz = matriz[:, :, 1]  # only looking at one dimension, 1 = green

    if debug:
        print('Image “',path,'“ opened with size:',im.size,'and mode:',im.mode)
        print('compressed the square-image with lenght :',
              oneD_matriz.shape[0], ' with factor:', cfactor)
        print('output matrix has shape:', oneD_matriz.shape)
        plt.imshow(oneD_matriz)
        plt.tight_layout()
        plt.show()

    return oneD_matriz

def center_leaf(path, square_side=2000):
    '''
    region we look at, because of the border we found with overlappingcenters a square on the leaf
   
    input: path of image
           square_side of matriz thats cut away    
    output: cut image
    '''
    up = 500
    down = 2900
    left = 400
    right = 4000
    s = square_side // 2
    
    im = Image.open(path).convert('RGB')
    matriz = np.array(im)  # convert image to numpy matrix    
    matriz ^= 0xFF  # invert matrix
    oneD_matriz = matriz[up:down,left:right,1] #only look at the green canal 1
    
    indices = np.argwhere(oneD_matriz >= 180) # give all pixel cordinates where the value is higer than 179
    meanx = np.average(indices[:,0]) + up
    meany = np.average(indices[:,1]) + left
    
    # select new area of the matrix, that is the input for CNN
    box = (meany - s, meanx - s, meany + s , meanx + s)
    new_image = im.crop(box)  # crop is Pill function
    im.close()
    return new_image

def center_leaf2_OUTDATED(path, square_side=2000, debug=False): #OUTDATED --> not use anymore
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
            total_weight_y += (oneD_matriz[x, y] >= 180)
            total_weight_x += (oneD_matriz[x, y] >= 180)


    # calculate the medium point to centralize the square around the medium
    # pixel (the square is the input of the CNN)
    meanx = 0
    meany = 0
    for x in range(up,down,5):  # only take evry 5 field for speed reasons -->makes code 25x faster
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
