import os
import gzip
import pickle
import data


class MNIST(data.Dataset):

    def __init__(self, data_file = 'mnist.pkl.gz',
                 #data_dir='/share/datasets/'):
                 data_dir = '/home/lucas/data/',
                 shape = (-1,784)):
        #############
        # LOAD DATA #
        #############

        data_path = os.path.join(data_dir,data_file)
        if (not os.path.isfile(data_path)) and data_file == 'mnist.pkl.gz':
            from six.moves import urllib
            origin = (
                'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
            )
            print('Downloading data from %s' % origin)
            urllib.request.urlretrieve(origin, data_path)

        print('... loading data')

        # Load the dataset
        with gzip.open(data_path, 'rb') as f:
            try:
                train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
            except:
                train_set, valid_set, test_set = pickle.load(f)
        # train_set, valid_set, test_set format: tuple(input, target)
        # input is a numpy.ndarray of 2 dimensions (a matrix)
        # where each row corresponds to an example. target is a
        # numpy.ndarray of 1 dimension (vector) that has the same length as
        # the number of rows in the input. It should give the target
        # to the example with the same index in the input.


        X_test, y_test = test_set
        X_valid, y_valid = valid_set
        X_train, y_train = train_set

        X_test  = data.floatX(X_test).reshape(shape)
        X_valid = data.floatX(X_valid).reshape(shape)
        X_train = data.floatX(X_train).reshape(shape)

        y_test = data.floatX(y_test)
        y_valid = data.floatX(y_valid)
        y_train = data.floatX(y_train)

        super(MNIST, self).__init__(X_train=X_train, y_train=y_train,
                                    X_valid=X_valid, y_valid=y_valid,
                                    X_test =X_test, y_test=y_test)

