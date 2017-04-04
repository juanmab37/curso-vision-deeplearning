import theano
import numpy as np

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def intX(X):
    return np.asarray(X, dtype=np.int32)

class Dataset(object):

    def __init__(self, X_train=None, y_train=None,
                 X_valid=None, y_valid=None,
                 X_test =None, y_test=None,
                 X_unlab=None):

        self.__dict__.update(locals())
        del self.self

    def get_train_batch(self, index, batch_size):
        X = self._get_x_batch(self.X_train,index,batch_size)
        y = self._get_y_batch(self.y_train,index,batch_size)
        return X, y

    def get_valid_batch(self, index, batch_size):
        X = self._get_x_batch(self.X_valid,index,batch_size)
        y = self._get_y_batch(self.y_valid,index,batch_size)
        return X, y

    def get_test_batch(self, index, batch_size):
        X = self._get_x_batch(self.X_test,index,batch_size)
        y = self._get_y_batch(self.y_test,index,batch_size)
        return X, y

    def get_unlab_batch(self, index, batch_size):
        X = self._get_x_batch(self.X_unlab,index,batch_size)
        return X

    def _get_x_batch(self, X, index, batch_size):
        size = X.shape[0]
        n1 = (index*batch_size)%size
        n2 = ((index+1)*batch_size-1)%size+1
        if n1>n2:
            return floatX(np.concatenate((X[n1:], X[:n2])))
        else:
            return floatX(X[n1:n2])


    def _get_y_batch(self, y, index, batch_size):
        size = y.shape[0]
        n1 = (index * batch_size) % size
        n2 = ((index + 1) * batch_size - 1) % size + 1
        if n1 > n2:
            return intX(np.concatenate((y[n1:], y[:n2])))
        else:
            return intX(y[n1:n2])
