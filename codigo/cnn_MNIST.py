import numpy
import theano
import theano.tensor as T
import argparse
import pickle

import lasagne

import mnist

"""HYPERPARAMS"""
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 500, help = 'Size of the minibatch')
parser.add_argument("--n_iter", type = int, default = 2000)
parser.add_argument("--lr", type = float, default = 0.01, help= 'Learning Rate')
parser.add_argument("--momentum", type = float, default = 0.9, help= 'Momentum')

parser.add_argument("--data_dir", default = '/home/lucas/data/', help= 'Directorio donde buscar el dataset')
parser.add_argument("--param_file", default = 'params.pkl')
hparams = parser.parse_args()
print hparams

#augmented_params =  dict( featurewise_center=True, featurewise_std_normalization=True,
#    rotation_range=20,
#    width_shift_range=0.2,
#    height_shift_range=0.2,
#    horizontal_flip=True,
#    dim_ordering='th',
#    )

"""DATASET"""
dataset = mnist.MNIST(data_dir=hparams.data_dir, shape=(-1,1,28,28)) #, augmented=augmented_params)


"""MODEL"""
# https://github.com/Lasagne/Lasagne/blob/master/examples/mnist.py
def build_cnn(input_var=None):
    # Input layer
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)

    # Convolutional layer with 32 kernels of size 5x5.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

X = T.tensor4('inputs')
network = build_cnn(X)
model_prob = lasagne.layers.get_output(network)
det_model_prob = lasagne.layers.get_output(network,deterministic=True)
det_model_out = T.argmax(det_model_prob, axis=1)


"""PARAMS"""
params = lasagne.layers.get_all_params(network, trainable=True)


"""LOSS"""
y = T.vector('y',dtype='int32')
loss = T.nnet.categorical_crossentropy(model_prob, y).mean()


"""OPTIMIZER"""
updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=hparams.lr, momentum=hparams.momentum)


"""TRAINING STEP FUNCTION"""
train_model = theano.function(
        inputs=[X,y],
        outputs=loss,
        updates=updates
)


"""MONITOR FUNCTIONS"""
predict = theano.function(
        inputs=[X],
        outputs=det_model_out,
        updates=None
)

"""TRANING MAIN LOOP"""
mon_frec = 10
valid_size = 10000
valid_batch = 500
best_error = 1.0
for it in xrange(hparams.n_iter):

    X_train, y_train = dataset.get_train_batch(hparams.batch_size)
    train_loss = train_model(X_train,y_train)

    """MONITOR"""
    if it % mon_frec == 0:
        valid_error = 0.0
        for valit in range(valid_size/valid_batch):
            X_valid, y_valid = dataset.get_valid_batch(valid_batch)
            y_pred = predict(X_valid)
            valid_error += (y_pred != y_valid).mean()
        valid_error /= valid_size/valid_batch
        y_pred = predict(X_train)
        train_error = (y_pred != y_train).mean()

        print it, train_loss, train_error, valid_error

        if best_error>valid_error:
            best_error = valid_error
            it_best = it
            values = lasagne.layers.get_all_param_values(network)
            with open(hparams.param_file, 'w') as f:
                pickle.dump(values, f)


"""PREDICTION"""
print 'Cargando el mejor modelo guardado'
values = pickle.load(open(hparams.param_file))

# Build the network and fill with pretrained weights
lasagne.layers.set_all_param_values(network, values)

valid_size = 10000
valid_batch = 500

valid_error = 0.0
for valit in range(valid_size / valid_batch):
    X_valid, y_valid = dataset.get_valid_batch(valid_batch)
    y_pred = predict(X_valid)
    valid_error += (y_pred != y_valid).mean()
valid_error /= valid_size / valid_batch

print it_best, valid_error
