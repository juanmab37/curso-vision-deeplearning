import numpy as np
import theano
import theano.tensor as T
import argparse
import pickle
from scipy.misc import imsave
from time import time

import lasagne
from lasagne.layers import InputLayer, DenseLayer, Conv2DLayer, \
    Deconv2DLayer, NonlinearityLayer, GlobalPoolLayer, DropoutLayer
from lasagne.layers import ReshapeLayer
from lasagne.layers.normalization import BatchNormLayer
from lasagne.nonlinearities import LeakyRectify, rectify, tanh, sigmoid, softmax
from lasagne.init import Normal
from lasagne.objectives import binary_crossentropy, categorical_crossentropy
from lasagne.regularization import regularize_layer_params, l2
from lasagne.updates import adam

import data
import stl10_data


from numpy.random import RandomState
np_rng = RandomState(1234)


"""HYPERPARAMS"""
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 100, help = 'Size of the minibatch')
parser.add_argument("--n_iter", type = int, default = 50000)
parser.add_argument("--lr", type = float, default = 0.0002, help= 'Learning Rate')
parser.add_argument("--gen_dim", type = int, default = 100, help= "Z's dimension")

parser.add_argument("--data_dir", default = '/home/lucas/data/stl10/', help= 'Directorio donde buscar el dataset')
parser.add_argument("--param_file", default = 'params.pkl')
hparams = parser.parse_args()
print hparams


"""DATASET"""
augmented_params =  dict(featurewise_center=False, featurewise_std_normalization=False,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    shear_range=0.3,
    zoom_range=(0.67,1),
    channel_shift_range=0.1,
    fill_mode='nearest',
    dim_ordering='th',
    )
dataset = stl10_data.stl10(data_dir=hparams.data_dir,augmented=augmented_params)

def center_crop(X, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = X.shape[2:4]
    if h == ph and w == pw:
        return X
    j = int(round((h - ph) / 2.))
    i = int(round((w - pw) / 2.))
    return X[:, :, j:j + ph, i:i + pw]


def random_crop(X, ph, pw=None):
    if pw is None:
        pw = ph
    h, w = X.shape[2:4]
    if h == ph and w == pw:
        return X
    j = np_rng.random_integers(0,h-ph)
    i = np_rng.random_integers(0,w-pw)
    return X[:, :, j:j + ph, i:i + pw]

def scale_data(X, x_min=0.0, x_max=255.0, new_min=-1.0, new_max=1.0):
        scale = x_max - x_min
        new_scale = new_max - new_min
        return data.floatX((X-x_min)*new_scale/scale+new_min)

def save_samples(X, (nh, nw), save_path=None):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img


"""MODEL"""
def build_generator(gen_dim=100, nchannels=3,
                    W_init=Normal(std=0.02)):

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

    c = DenseLayer(DropoutLayer(h, p=.5), num_units=nclasses, nonlinearity=softmax)

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

weight_decay = regularize_layer_params(classifier, l2)
class_loss = categorical_crossentropy(classX,targets).mean() + 0.01 * weight_decay


"""OPTIMIZER"""
dis_updates = adam(dis_loss,discrim_params,
                   learning_rate=hparams.lr, beta1=0.5, beta2=0.999)
gen_updates = adam(gen_loss,gen_params,
                   learning_rate=hparams.lr, beta1=0.5, beta2=0.999)
class_updates = adam(class_loss,classif_params,learning_rate=hparams.lr)



"""TRAINING STEP FUNCTION"""
print "Compiling training functions"
train_gen = theano.function([Z], gen_loss, updates=gen_updates)
train_dis = theano.function([X, Z], dis_loss, updates=dis_updates)
train_class = theano.function([X, targets], class_loss, updates=class_updates)




"""MONITOR FUNCTIONS"""
print "Compiling monitor functions"

gen = theano.function([Z], genX)

predict = theano.function(
        inputs=[X],
        outputs=classXTest_out,
        updates=None
)

"""TRANING MAIN LOOP"""
mon_frec = 100
valid_size = 200
valid_batch = 50
best_acc = 0.0
Z_sample = data.floatX(np_rng.uniform(-1., 1., size=(10*10, hparams.gen_dim)))
last_it = 0
it_best = 0
t = time()
for it in xrange(hparams.n_iter+1):

    Z_batch = data.floatX(np_rng.uniform(-1., 1., size=(hparams.batch_size, hparams.gen_dim)))
    X_batch = dataset.get_unlab_batch(hparams.batch_size)
    X_batch = scale_data(random_crop(X_batch, 64))

    dloss = train_dis(X_batch, Z_batch)
    gloss = train_gen(Z_batch)

    X_batch,y_batch = dataset.get_train_batch(hparams.batch_size)
    X_batch = scale_data(center_crop(X_batch, 64))
    closs = train_class(X_batch,y_batch)


    """MONITOR"""
    if it % mon_frec == 0:
        valid_acc = 0.0
        for valit in range(valid_size/valid_batch):
            X_valid, y_valid = dataset.get_valid_batch(valid_batch)
            X_valid = scale_data(center_crop(X_valid, 64))
            y_pred = predict(X_valid)
            valid_acc += (y_pred == y_valid).mean()
        valid_acc /= valid_size/valid_batch
        y_pred = predict(X_batch)
        train_acc = (y_pred == y_batch).mean()


        if best_acc<valid_acc:
            best_acc = valid_acc
            it_best = it
            values = lasagne.layers.get_all_param_values(classifier)
            with open(hparams.param_file, 'w') as f:
                pickle.dump(values, f)

        samples = np.asarray(gen(Z_sample))
        save_samples(scale_data(samples,-1.0,1.0,0,255).transpose(0, 2, 3, 1), (10, 10), 'samples_%05d.png'%it)

        print it, dloss, gloss, closs, train_acc, valid_acc, best_acc, it_best

        with open('monitor.log', 'a') as f:
            np.savetxt(f, [[it, dloss, gloss, closs, train_acc, valid_acc]], fmt='%1.3e')

        t2 = time()-t
        t += t2
        horas = t2/(1+it-last_it)/3600.*10000
        print "iter:%d/%d;  %4.2f horas. para 10000 iteraciones"%(it+1,hparams.n_iter,horas)
        last_it = it+1

"""PREDICTION"""
print 'Cargando el mejor modelo guardado'
values = pickle.load(open(hparams.param_file))

# Build the network and fill with pretrained weights
lasagne.layers.set_all_param_values(classifier, values)

valid_acc = 0.0
for valit in range(valid_size / valid_batch):
    X_valid, y_valid = dataset.get_valid_batch(valid_batch)
    X_valid = scale_data(center_crop(X_valid, 64))
    y_pred = predict(X_valid)
    valid_acc += (y_pred == y_valid).mean()
valid_acc /= valid_size / valid_batch

print it_best, valid_acc