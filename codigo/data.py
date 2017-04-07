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
                 X_unlab=None, augmented=None):

        self.index = 0
        self.augmented = augmented
        if self.augmented:
            from keras.preprocessing.image import ImageDataGenerator
            self.datagen = ImageDataGenerator(**self.augmented)
            self.datagen.fit(X_train)

        self.__dict__.update(locals())
        del self.self

    def get_train_batch(self, batch_size):
        X,y = self._get_xy_batch(self.X_train, self.y_train, batch_size)
        #y = self._get_y_batch(self.y_train,index,batch_size)
        return X, y

    def get_valid_batch(self, batch_size):
        X,y = self._get_xy_batch(self.X_valid, self.y_valid, batch_size)
        #y = self._get_y_batch(self.y_valid,batch_size)
        return X, y

    def get_test_batch(self, batch_size):
        X = self._get_xy_batch(self.X_test, self.y_test, batch_size)
        #y = self._get_y_batch(self.y_test,batch_size)
        return X, y

    def get_unlab_batch(self, batch_size):
        X = self._get_x_batch(self.X_unlab,batch_size)
        return X

    def _get_x_batch(self, X, batch_size):

        if self.augmented:
            X,_ = self.datagen.flow(X, [0]*batch_size, batch_size=batch_size).next()
            return floatX(X)

        size = X.shape[0]
        n1 = (self.index*batch_size)%size
        n2 = ((self.index+1)*batch_size-1)%size+1

        self.index = self.index + 1
        if n1>n2:
            return floatX(np.concatenate((X[n1:], X[:n2])))
        else:
            return floatX(X[n1:n2])

    def _get_xy_batch(self, X,y, batch_size):

        if self.augmented:
            X,y = self.datagen.flow(X, y, batch_size=batch_size).next()
            return floatX(X),intX(y)

        size = X.shape[0]
        n1 = (self.index*batch_size)%size
        n2 = ((self.index+1)*batch_size-1)%size+1

        self.index = self.index + 1
        if n1>n2:
            return floatX(np.concatenate((X[n1:], X[:n2]))), intX(np.concatenate((y[n1:], y[:n2])))

        else:
            return floatX(X[n1:n2]), intX(y[n1:n2])
