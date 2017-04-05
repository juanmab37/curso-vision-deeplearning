import h5py
import numpy as np
from collections import defaultdict

import data

class HDF5Matrix():
    refs = defaultdict(int)

    def __init__(self, datapath, dataset, start=None, end=None, normalizer=None):
        if datapath not in list(self.refs.keys()):
            f = h5py.File(datapath)
            self.refs[datapath] = f
        else:
            f = self.refs[datapath]

        self.data = f[dataset]

        if (start == None and end == None):
            self.start = 0
            self.end = self.data.shape[0]
        else:
            assert(start <> None and end <> None)
            self.start = start
            self.end = end

        self.normalizer = normalizer

    def __len__(self):
        return self.end - self.start

    def __getitem__(self, key):
        if isinstance(key, slice):
            if key.stop + self.start <= self.end:
                idx = slice(key.start+self.start, key.stop + self.start)
            else:
                raise IndexError
        elif isinstance(key, int):
            if key + self.start < self.end:
                idx = key+self.start
            else:
                raise IndexError
        elif isinstance(key, np.ndarray):
            if np.max(key) + self.start < self.end:
                idx = (self.start + key).tolist()
            else:
                raise IndexError
        elif isinstance(key, list):
            if max(key) + self.start < self.end:
                idx = [x + self.start for x in key]
            else:
                raise IndexError
        if self.normalizer is not None:
            return self.normalizer(self.data[idx])
        else:
            return self.data[idx]

    @property
    def shape(self):
        return self.data.shape
        #return tuple([self.end - self.start, self.data.shape[1]])

# example:
# X_train = HDF5Matrix('dataset.h5', 'X', start=0, end=200) 
# y_train = HDF5Matrix('dataset.h5', 'y', start=0, end=200) 

class HDF5Dataset(data.Dataset):
    def __init__(self, train_file, valid_file, test_file):

        X_train = HDF5Matrix(train_file, 'X')
        y_train = HDF5Matrix(train_file, 'y')

        X_valid = HDF5Matrix(valid_file, 'X')
        y_valid = HDF5Matrix(valid_file, 'y')

        X_test = HDF5Matrix(test_file, 'X')
        y_test = HDF5Matrix(test_file, 'y')

        self.shape = X_train.data.shape

        super(HDF5Dataset, self).__init__(X_train=X_train, y_train=y_train,
                                          X_valid=X_valid, y_valid=y_valid,
                                          X_test =X_test, y_test=y_test)
