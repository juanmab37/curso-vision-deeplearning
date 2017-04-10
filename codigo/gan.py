import numpy
import theano
import theano.tensor as T
import argparse
import pickle

import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, Deconv2DLayer, NonlinearityLayer, GlobalPoolLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import LeakyRectify, rectify, tanh, sigmoid, softmax
from lasagne.init import Normal
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.updates import adam

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


"""DATASET"""
dataset = mnist.MNIST(data_dir=hparams.data_dir, shape=(-1,1,28,28))


"""MODEL"""
def build_generator(gen_dim=100, nchannels=3,
                    W_init=Normal(std=0.02)):
    print 'gen'

    Z = InputLayer((None, gen_dim))

    ### LAYER 1
    h = DenseLayer(Z, num_units=128 * 8 * 4 * 4, W=W_init, b=None, name='gen_l1_Dense')
    h = BatchNormLayer(h, epsilon=1e-4, name='gen_l1_BN')
    h = NonlinearityLayer(h, nonlinearity=rectify, name='gen_l1_relu')
    h = ReshapeLayer(h, ([0], 128 * 8, 4, 4))

    ### LAYER 2
    x1 = Deconv2DLayer(h, num_filters=128 * 4,
                       filter_size=5, stride=2,
                       crop='same', W=W_init, b=None, nonlinearity=None,
                       output_size=[8,8], name='gen_l2_Deconv')
    x2 = BatchNormLayer(x1, epsilon=1e-4, name='gen_l2_BN')
    h = NonlinearityLayer(x2, nonlinearity=rectify, name='gen_l2_relu')

    ### LAYER 3
    x1 = Deconv2DLayer(h, num_filters=128 * 2,
                       filter_size=5, stride=2,
                       crop='same', W=W_init, b=None, nonlinearity=None,
                       output_size=[16,16], name='gen_l3_Deconv')
    x2 = BatchNormLayer(x1, epsilon=1e-4, name='gen_l3_BN')
    h = NonlinearityLayer(x2, nonlinearity=rectify, name='gen_l3_relu')

    ### LAYER 4
    x1 = Deconv2DLayer(h, num_filters=128 * 1,
                       filter_size=5, stride=2,
                       crop='same', W=W_init, b=None, nonlinearity=None,
                       output_size=[32,32], name='gen_l4_Deconv')
    x2 = BatchNormLayer(x1, epsilon=1e-4, name='gen_l4_BN')
    h = NonlinearityLayer(x2, nonlinearity=rectify, name='gen_l4_relu')

    ### LAYER 5
    x1 = Deconv2DLayer(h, num_filters=nchannels,
                       filter_size=5, stride=2,
                       crop='same', W=W_init, b=None, nonlinearity=None,
                       output_size=[64,64], name='gen_l5_Deconv')
    x = NonlinearityLayer(x1, nonlinearity=tanh, name='gen_l5_tanh')

    return x


def build_discriminator(nchannels=3, W_init=Normal(std=0.02), nclasses = 10):

    X = InputLayer((None, nchannels, 64, 64))

    ### LAYER 1
    x1 = Conv2DLayer(X, num_filters=128, filter_size=5, stride=2, pad='same',
                     W=W_init, b=None, nonlinearity=None, name='dis_l1_Conv')
    x2 = BatchNormLayer(x1, epsilon=1e-4, name='dis_l1_BN')
    h = NonlinearityLayer(x2, nonlinearity=LeakyRectify(0.2), name='dis_l1_lrelu')
    
    # for a,i in zip([2,4,8],[2,3,4]):
    ### LAYER 2
    x1 = Conv2DLayer(h, num_filters=2 * 128, filter_size=5, stride=2, pad='same',
                     W=W_init, b=None, nonlinearity=None, name='dis_l2_Conv')
    x2 = BatchNormLayer(x1, epsilon=1e-4, name='dis_l2_BN')
    h = NonlinearityLayer(x2, nonlinearity=LeakyRectify(0.2))

    ### LAYER 3
    x1 = Conv2DLayer(h, num_filters=4 * 128, filter_size=5, stride=2, pad='same',
                     W=W_init, b=None, nonlinearity=None, name='dis_l3_Conv')
    x2 = BatchNormLayer(x1, epsilon=1e-4, name='dis_l3_BN')
    h = NonlinearityLayer(x2, nonlinearity=LeakyRectify(0.2))

    ### LAYER 4
    x1 = Conv2DLayer(h, num_filters=8 * 128, filter_size=5, stride=2, pad='same',
                     W=W_init, b=None, nonlinearity=None, name='dis_l4_Conv')
    x2 = BatchNormLayer(x1, epsilon=1e-4, name='dis_l4_BN')
    h = NonlinearityLayer(x2, nonlinearity=LeakyRectify(0.2))

    ### LAYER 1
    h = GlobalPoolLayer(h, pool_function=T.max)
    # h_shape = lasagne.layers.get_output_shape(h)
    # wsig = theano.shared(W_init.sample((h_shape[1], 1)))
    y = DenseLayer(h, num_units=1, W=W_init, b=None, nonlinearity=sigmoid, name='dis_l5_DenseSigmoid')

    c = DenseLayer(h, num_units=nclasses, nonlinearity=softmax)

    return y, c


X = T.tensor4()
dis_network, classifier = build_discriminator(nchannels=3, nclasses=10)
disX = lasagne.layers.get_output(dis_network, X)
disXTest = lasagne.layers.get_output(dis_network, X, deterministic=True)


classX = lasagne.layers.get_output(classifier, X)
classXTest = lasagne.layers.get_output(classifier, X, deterministic=True)
classXTest_out = T.argmax(classXTest, axis=1)

Z = T.matrix()
gen_network = build_generator(hparams.gen_dim, nchannels=3)
genX = lasagne.layers.get_output(gen_network, Z)
disgenX = lasagne.layers.get_output(dis_network, genX)
genXTest = lasagne.layers.get_output(gen_network, Z, deterministic=True)
disgenXTest = lasagne.layers.get_output(dis_network, genXTest, deterministic=True)



"""PARAMS"""
discrim_params = lasagne.layers.get_all_params(dis_network, trainable=True)
classif_params = lasagne.layers.get_all_params(classifier, trainable=True)
gen_params     = lasagne.layers.get_all_params(gen_network, trainable=True)



"""LOSS"""
targets = T.vector('y',dtype='int32')
dis_loss = binary_crossentropy(disX, T.ones(disX.shape)).mean() +\
            binary_crossentropy(disgenX, T.zeros(disgenX.shape)).mean()
gen_loss = binary_crossentropy(disgenX, T.ones(disgenX.shape)).mean()
class_loss = categorical_crossentropy(classX,targets).mean()


"""OPTIMIZER"""
dis_updates = adam(dis_loss,discrim_params,
                   learning_rate=0.0002, beta1=0.5, beta2=0.999)
gen_updates = adam(gen_loss,gen_params,
                   learning_rate=0.0002, beta1=0.5, beta2=0.999)
class_updates = adam(class_loss,classif_params,learning_rate=0.0002)



"""TRAINING STEP FUNCTION"""
train_gen = theano.function([Z], gen_loss, updates=gen_updates)
train_dis = theano.function([X, Z], dis_loss, updates=dis_updates)
train_class = theano.function([X, targets], class_loss, updates=class_updates)




"""MONITOR FUNCTIONS"""
gen = theano.function([Z], genX)

predict = theano.function(
        inputs=[X],
        outputs=classXTest_out,
        updates=None
)

"""TRANING MAIN LOOP"""
mon_frec = 10
valid_size = 10000
valid_batch = 500
best_error = 1.0
for it in xrange(hparams.n_iter):

    X_train, y_train = dataset.get_train_batch(it,hparams.batch_size)
    train_loss = train_model(X_train,y_train)

    """MONITOR"""
    if it % mon_frec == 0:
        valid_error = 0.0
        for valit in range(valid_size/valid_batch):
            X_valid, y_valid = dataset.get_valid_batch(valit, valid_batch)
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
    X_valid, y_valid = dataset.get_valid_batch(valit, valid_batch)
    y_pred = predict(X_valid)
    valid_error += (y_pred != y_valid).mean()
valid_error /= valid_size / valid_batch

print it_best, valid_error