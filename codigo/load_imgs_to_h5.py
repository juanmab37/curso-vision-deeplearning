#!/usr/bin/python2

import os
import numpy as np

from PIL import Image as pil_image
from sys import argv

def img_to_array(img, data_format=None):
    """Converts a PIL Image instance to a Numpy array.
    # Arguments
        img: PIL Image instance.
        data_format: Image data format.
    # Returns
        A 3D Numpy array.
    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format is None:
        data_format = 'channels_last'
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ', data_format)
    # Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but original PIL image has format (width, height, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if data_format == 'channels_first':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if data_format == 'channels_first':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise ValueError('Unsupported image shape: ', x.shape)
    return x


def load_img(path, grayscale=False, target_size=None):
    """Loads an image into PIL format.
    # Arguments
        path: Path to image file
        grayscale: Boolean, whether to load the image as grayscale.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
    # Returns
        A PIL Image instance.
    # Raises
        ImportError: if PIL is not available.
    """
    if pil_image is None:
        raise ImportError('Could not import PIL.Image. '
                          'The use of `array_to_img` requires PIL.')
    img = pil_image.open(path)
    if grayscale:
        if img.mode != 'L':
            img = img.convert('L')
    else:
        if img.mode != 'RGB':
            img = img.convert('RGB')
    if target_size:
        wh_tuple = (target_size[1], target_size[0])
        if img.size != wh_tuple:
            img = img.resize(wh_tuple)
    return img


def list_pictures(directory, ext='.png'):
    return [os.path.join(root, f)
            for root, _, files in os.walk(directory) for f in files
            if ext in f]

def create_dataset(filename, X, y, size):
    assert(len(X) == len(y))
    import h5py
    f = h5py.File(filename, 'w')
    X_size = (len(X), size[0], size[1], size[2])
    X_dset = f.create_dataset('X', X_size, dtype='f')
    X_dset[:] = X
    y_size = (len(y),)
    y_dset = f.create_dataset('y', y_size, dtype='i')
    y_dset[:] = y
    f.close()


if len(argv) <> 3:
    print "USAGE: dir filename"

else:

    imgs = []
    dirs = dict()
    labels = []
    i = 0


    input_dir = argv[1]
    h5_filename = argv[2]

    for filename in list_pictures(argv[1]):
        img = load_img(filename)
        imgs.append(img_to_array(img))
        label = filename.replace(input_dir,"")
        label = "/".join(label.split("/")[:-1])
        
        if not(label in dirs):
            dirs[label] = i
            i = i + 1
       
        labels.append(dirs[label])

    print imgs[0].shape
    print dirs
    create_dataset(h5_filename, imgs, labels, imgs[0].shape)
