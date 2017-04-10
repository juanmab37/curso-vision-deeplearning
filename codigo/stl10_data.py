import os, sys, tarfile, urllib
import numpy as np

import data

class stl10(data.Dataset):

    def __init__(self, train_fold=0,
             n_valid_folds=5,
             valid_fold=0,
             train = True,
             valid = True,
             test = True,
             unlab = True,
             data_dir='/share/datasets/stl10',
             data_url='http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
             ):

        height = 96
        width = 96
        depth = 3

        # size of a single image in bytes
        dim = height * width * depth

        self.__dict__.update(locals())
        del self.self # We don't need this either

        x_min= 0
        x_max = 255

        if train:
            X_train, y_train = self.load_data(data_dir,'train')
            if train_fold is not None:
                ifold = self.load_fold_indices(data_dir,train_fold)
                X_train = X_train[ifold]
                y_train = y_train[ifold]
            self.train_size = X_train.shape[0] # 1000 
        else:
            X_train = None
            y_train = None

        if test:
            X_test, y_test = self.load_data(data_dir,'test')
            self.test_size = X_test.shape[0]
        else:
            X_test = None
            y_test = None

        if unlab:
            X_unlab, y_unlab = self.load_data(data_dir,'unlabeled')
            self.unlab_size = X_unlab.shape[0] # 100000
        else:
            X_unlab = None

        if valid:
            if train:
                self.valid_size = self.train_size / n_valid_folds
                valid_mask = np.zeros(self.train_size, dtype=bool)
                valid_mask[valid_fold*self.valid_size:(valid_fold+1)*self.valid_size] = True
                X_valid = X_train[valid_mask]
                y_valid = y_train[valid_mask]
                X_train = X_train[-valid_mask]
                y_train = y_train[-valid_mask]
                self.train_size = X_train.shape[0]
            elif unlab:
                self.valid_size = self.unlab_size / n_valid_folds
                valid_mask = np.zeros(self.unlab_size, dtype=bool)
                valid_mask[valid_fold*self.valid_size:(valid_fold+1)*self.valid_size] = True
                X_valid = X_unlab[valid_mask]
                y_valid = None
                X_unlab = X_unlab[-valid_mask]
                self.unlab_size = X_unlab.shape[0]
        else:
            X_valid = None
            y_valid = None

        super(stl10, self).__init__(X_train=X_train, y_train=y_train,
                                    X_valid=X_valid, y_valid=y_valid,
                                    X_test =X_test, y_test=y_test,
                                    X_unlab=X_unlab)

    def load_data(self, data_dir, which_set):

        if which_set == 'train' or which_set == 'test':
            path_to_data  = os.path.join(data_dir, 'stl10_binary/' + which_set + '_X.bin')
            path_to_labels= os.path.join(data_dir, 'stl10_binary/' + which_set + '_y.bin')
        elif which_set == 'unlabeled':
            path_to_data =  os.path.join(data_dir, 'stl10_binary/' + which_set + '_X.bin')
        else:
            raise Exception('no existe ' + which_set)

        y = None
        if which_set is not "unlabeled":
            with open(path_to_labels, 'rb') as f:
                y = np.fromfile(f, dtype=np.uint8) - np.uint8(1)

        with open(path_to_data, 'rb') as f:
            X = np.fromfile(f, dtype=np.uint8)

        X = np.reshape(X, (-1, 3, 96, 96))
        X = np.transpose(X, (0, 1, 3, 2))

        return X, y

    def load_fold_indices(self, data_dir, train_fold):
        ifold = np.loadtxt(os.path.join(data_dir, 'stl10_binary/fold_indices.txt'), dtype=int)
        return ifold[train_fold]

    def download_and_extract(self, data_dir, data_url):
        """
        Download and extract the STL-10 dataset
        :return: None
        """
        dest_directory = data_dir
        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
            filepath, _ = urllib.urlretrieve(data_url, filepath, reporthook=_progress)
            print('Downloaded', filename)
            tarfile.open(filepath, 'r:gz').extractall(dest_directory)

