"""Read HDF5 file containing the 70000 MNIST handwritten digits."""

import h5py
import numpy as np


def main():
    hdf = h5py.File('data/mnist-original.h5', mode='r')

    list_of_datasets = []

    def func(name, obj):
        if isinstance(obj, h5py.Dataset):
            list_of_datasets.append((name, obj))
    hdf.visititems(func)

    data = list_of_datasets[0][1]
    label = list_of_datasets[1][1]
    ordering = list_of_datasets[2][1]

    all_images = np.zeros(data.shape, dtype=data.dtype)
    all_labels = 10 * np.ones(label.shape, dtype=label.dtype)
    all_orderings = np.zeros(ordering.shape, dtype=ordering.dtype)
    swap_shape = (data.shape[1], data.shape[0])
    swapped_images = np.zeros(swap_shape, dtype=data.dtype)

    data.read_direct(all_images)
    label.read_direct(all_labels)
    ordering.read_direct(all_orderings)
    swapped_images = np.copy(all_images.swapaxes(0, 1))
    swapped_images = swapped_images.astype(
        np.float32) / 256.0  # normalization [0,1)

    np.savez_compressed(
        'data/MNIST_academia',
        Images=swapped_images,
        Labels=all_labels)
    ''' # Reload the data
    loaded_file = np.load('data/MNIST_academia.npz')
    images = load['Images']
    labels = load['Labels']
    '''


if __name__ == "__main__":
    main()
