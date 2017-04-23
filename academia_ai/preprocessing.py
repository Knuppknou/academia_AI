import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from .leafs import leafs
print("Loaded preprocessing!")

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
    '''
    Takes as Input the matrices from collectData and creates a training and a testing list'''
    l_train = []
    l_test = []

    for n in range (7):
        matrices = np.load(os.path.join(directory, str(n)+'.npy'))
        for i in range(759): # 2x800 for training
            l_train += [leafs.Leaf(i+n*1000, n, matrices[i]/255)]
        for i in range(760,839): # 2x80 for testing
            l_test += [leafs.Leaf(i+n*1000, n, matrices[i]/255)]
    
    if shuffle: 
        np.random.shuffle(l_train)
        np.random.shuffle(l_test)
        
    return([l_train,l_test])
    
        
def collectData(root_path, save_path, cfactor, overwrite = False):
    '''processes images from root_path one-by-one and save them in same directory
    collect them tree by tree, set their labels and return a training and a testing list'''
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
   
    input: path of image square_side of matriz thats cut away    
    output: cut image
    ATTENTION: the cutting borders are fixed
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


def find_overlap(root_path):
    '''function to overlap all pictures
    creates a image of all overlayed pictures so the interesting area of the picture can manually be classified'''
    maximum = np.zeros((3456, 4608)) #
    for root, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            im_path = (os.path.join(root, name))
            if name[0] == 'I':  #making sure its an image, because there are some other files in the directory
                image = Image.open(im_path)
                image.convert('RGB')
                matriz = np.array(image)
                
                maximum = np.maximum(maximum, matriz[:, :, 0])
                maximum = np.maximum(maximum, matriz[:, :, 1])
                maximum = np.maximum(maximum, matriz[:, :, 2])
                image.close()
    return maximum
