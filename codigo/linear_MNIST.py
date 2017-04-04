import numpy
import theano
import theano.tensor as T
import argparse

import mnist

"""HYPERPARAMS"""
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 600, help = 'Size of the minibatch')
parser.add_argument("--n_iter", type = int, default = 50000)
parser.add_argument("--init_scale", type = float, default = 1e-6, help = 'Weights init scale')
parser.add_argument("--lr", type = float, default = 0.13, help= 'Learning Rate')
hparams = parser.parse_args()
print hparams


"""DATASET"""
dataset = mnist.MNIST()


"""PARAMS"""
n_in = 28*28
n_out = 10
# initialize the weights W as a matrix of shape (n_in, n_out)
W = theano.shared(
        value=numpy.random.normal(scale=hparams.init_scale,size=(n_in, n_out)).astype(theano.config.floatX),
        name='W'
)
# initialize the biases b as a vector of n_out 0s
b = theano.shared(
    value=numpy.zeros((n_out,),dtype=theano.config.floatX),
    name='b'
)
params = [W, b] # Lista de todos los parametros del modelo


"""MODEL"""
X = T.matrix('X')  # data, presented as rasterized images
model_prob = T.nnet.softmax(T.dot(X, W) + b)
model_out = T.argmax(model_prob, axis=1)


"""LOSS"""
y = T.vector('y',dtype='int32')
loss = T.nnet.categorical_crossentropy(model_prob, y).mean()


"""OPTIMIZER"""
# SGD
# compute the gradient of cost with respect to theta = (W,b)
g_W = T.grad(cost=loss, wrt=W)
g_b = T.grad(cost=loss, wrt=b)

# specify how to update the parameters of the model as a list of
# (variable, update expression) pairs.
updates = [(W, W - hparams.lr * g_W),
           (b, b - hparams.lr * g_b)]


"""TRAINING STEP FUNCTION"""
train_model = theano.function(
        inputs=[X,y],
        outputs=loss,
        updates=updates
)


"""MONITOR FUNCTIONS"""
valid_model = theano.function(
        inputs=[X,y],
        outputs=loss,
        updates=None
)
predict = theano.function(
        inputs=[X],
        outputs=model_out,
        updates=None
)

"""TRANING MAIN LOOP"""
mon_frec = 1000
for it in xrange(hparams.n_iter):

    X_train, y_train = dataset.get_train_batch(it,hparams.batch_size)
    train_loss = train_model(X_train,y_train)

    """MONITOR"""
    if it % mon_frec == 0:
        X_valid, y_valid = dataset.get_valid_batch(it/mon_frec,hparams.batch_size)
        valid_loss = valid_model(X_valid,y_valid)
        y_pred = predict(X_valid)
        valid_error = (y_pred != y_valid).mean()
        y_pred = predict(X_train)
        train_error = (y_pred != y_train).mean()

        print it, train_loss, valid_loss, train_error, valid_error
