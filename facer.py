#!/usr/bin/env python3
"""                          _              _
  _ __   ___ _   _ _ __ __ _| |   ___ _ __ | |__   __ _ _ __   ___ ___
 | '_ \ / _ \ | | | '__/ _` | |  / _ \ '_ \| '_ \ / _` | '_ \ / __/ _ \
 | | | |  __/ |_| | | | (_| | | |  __/ | | | | | | (_| | | | | (_|  __/
 |_| |_|\___|\__,_|_|  \__,_|_|  \___|_| |_|_| |_|\__,_|_| |_|\___\___|


      a version of neural-enhance without training functionality
                made to be more lean and efficient

"""
#
# Copyright (c) 2016, Alex J. Champandard.
#
# Neural Enhance is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General
# Public License version 3. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#

__version__ = '0.3'

import io
import os
import sys
import bz2
import glob
import math
import time
import dlib
import pickle
import random
import argparse
import itertools
import threading
import collections
import face_recognition


# Configure all options first so we can later custom-load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument
add_arg('files',                nargs='*', default=[])
add_arg('--zoom',               default=2, type=int,                help='Resolution increase factor for inference.')
add_arg('--rendering-tile',     default=80, type=int,               help='Size of tiles used for rendering images.')
add_arg('--rendering-overlap',  default=24, type=int,               help='Number of pixels padding around each tile.')
add_arg('--rendering-histogram',default=False, action='store_true', help='Match color histogram of output to input.')
add_arg('--type',               default='photo', type=str,          help='Name of the neural network to load/save.')
add_arg('--model',              default='default', type=str,        help='Specific trained version of the model.')
add_arg('--train',              default=False, type=str,            help='File pattern to load for training.')
add_arg('--train-scales',       default=0, type=int,                help='Randomly resize images this many times.')
add_arg('--train-blur',         default=None, type=int,             help='Sigma value for gaussian blur preprocess.')
add_arg('--train-noise',        default=None, type=float,           help='Radius for preprocessing gaussian blur.')
add_arg('--train-jpeg',         default=[], nargs='+', type=int,    help='JPEG compression level & range in preproc.')
add_arg('--epochs',             default=10, type=int,               help='Total number of iterations in training.')
add_arg('--epoch-size',         default=72, type=int,               help='Number of batches trained in an epoch.')
add_arg('--save-every',         default=10, type=int,               help='Save generator after every training epoch.')
add_arg('--batch-shape',        default=192, type=int,              help='Resolution of images in training batch.')
add_arg('--batch-size',         default=15, type=int,               help='Number of images per training batch.')
add_arg('--buffer-size',        default=1500, type=int,             help='Total image fragments kept in cache.')
add_arg('--buffer-fraction',    default=5, type=int,                help='Fragments cached for each image loaded.')
add_arg('--learning-rate',      default=1E-4, type=float,           help='Parameter for the ADAM optimizer.')
add_arg('--learning-period',    default=75, type=int,               help='How often to decay the learning rate.')
add_arg('--learning-decay',     default=0.5, type=float,            help='How much to decay the learning rate.')
add_arg('--generator-upscale',  default=2, type=int,                help='Steps of 2x up-sampling as post-process.')
add_arg('--generator-downscale',default=0, type=int,                help='Steps of 2x down-sampling as preprocess.')
add_arg('--generator-filters',  default=[64], nargs='+', type=int,  help='Number of convolution units in network.')
add_arg('--generator-blocks',   default=4, type=int,                help='Number of residual blocks per iteration.')
add_arg('--generator-residual', default=2, type=int,                help='Number of layers in a residual block.')
add_arg('--perceptual-layer',   default='conv2_2', type=str,        help='Which VGG layer to use as loss component.')
add_arg('--perceptual-weight',  default=1e0, type=float,            help='Weight for VGG-layer perceptual loss.')
add_arg('--discriminator-size', default=32, type=int,               help='Multiplier for number of filters in D.')
add_arg('--smoothness-weight',  default=2e5, type=float,            help='Weight of the total-variation loss.')
add_arg('--adversary-weight',   default=5e2, type=float,            help='Weight of adversarial loss compoment.')
add_arg('--generator-start',    default=0, type=int,                help='Epoch count to start training generator.')
add_arg('--discriminator-start',default=1, type=int,                help='Epoch count to update the discriminator.')
add_arg('--adversarial-start',  default=2, type=int,                help='Epoch for generator to use discriminator.')
add_arg('--device',             default='cpu', type=str,            help='Name of the CPU/GPU to use, for Theano.')
args = parser.parse_args()


#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information a little better, plus it looks cool!
class ansi:
    WHITE = '\033[0;97m'
    WHITE_B = '\033[1;97m'
    YELLOW = '\033[0;33m'
    YELLOW_B = '\033[1;33m'
    RED = '\033[0;31m'
    RED_B = '\033[1;31m'
    BLUE = '\033[0;94m'
    BLUE_B = '\033[1;94m'
    CYAN = '\033[0;36m'
    CYAN_B = '\033[1;36m'
    ENDC = '\033[0m'

def error(message, *lines):
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + ("{}\n" if lines else "{}")
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    sys.exit(-1)

def warn(message, *lines):
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    print(string.format(ansi.YELLOW_B, ansi.YELLOW, ansi.ENDC))

# itertools.chain turns multiple arrays into single array
def extend(lst): return itertools.chain(lst, itertools.repeat(lst[-1]))

print("""{}   {}Super Resolution for images and videos powered by Deep Learning!{}
  - Code licensed as AGPLv3, models under CC BY-NC-SA.{}""".format(ansi.CYAN_B, __doc__, ansi.CYAN, ansi.ENDC))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,'\
                                      'print_active_device=False'.format(args.device))

# Scientific & Imaging Libraries
import numpy as np
import scipy.ndimage, scipy.misc, PIL.Image

# Numeric Computing (GPU)
import theano, theano.tensor as T
T.nnet.softminus = lambda x: x - T.nnet.softplus(x)

# Support ansi colors in Windows too.
if sys.platform == 'win32':
    import colorama

# Deep Learning Framework
import lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
from lasagne.layers import InputLayer, ConcatLayer, ElemwiseSumLayer, batch_norm

print('{}  - Using the device `{}` for neural computation.{}\n'.format(ansi.CYAN, theano.config.device, ansi.ENDC))


#======================================================================================================================
# Convolution Networks
#======================================================================================================================

class SubpixelReshuffleLayer(lasagne.layers.Layer):
    """Based on the code by ajbrock: https://github.com/ajbrock/Neural-Photo-Editor/
    """
    # inherets lasagne.layers.Layer

    def __init__(self, incoming, channels, upscale, **kwargs):
        super(SubpixelReshuffleLayer, self).__init__(incoming, **kwargs)
        # incoming == layer, channels is , upscale is size of change
        self.upscale = upscale
        self.channels = channels

    def get_output_shape_for(self, input_shape):
        def up(d): return self.upscale * d if d else d
        return (input_shape[0], self.channels, up(input_shape[2]), up(input_shape[3]))

    def get_output_for(self, input, deterministic=False, **kwargs):
        out, r = T.zeros(self.get_output_shape_for(input.shape)), self.upscale
        for y, x in itertools.product(range(r), repeat=2):
            out=T.inc_subtensor(out[:,:,y::r,x::r], input[:,r*y+x::r*r,:,:])
        return out


# class for the Model which will be 'default' in this version
class Model(object):

    # Model constructor
    def __init__(self):
        # the model class is partially a wrapper
        self.network = collections.OrderedDict()
        # InputLayer represents network input
        # first arg is dimension of element
        # represents the image placed in OrderedDict
        # InputLayer is lasagne function
        self.network['img'] = InputLayer((None, 3, None, None))
        # put seed of img in OrderedDict
        self.network['seed'] = InputLayer((None, 3, None, None))
        # loads neural network model from bz2 file
        # configs is a dict with generator values
        # such as filter, residual, upscale (zoom) value
        # params keys of network keys and np arrays
        config, params = self.load_model()
        # create generator
        self.setup_generator(self.last_layer(), config)
        # load generator
        self.load_generator(params)
        # calls function to generate callable objects from graphs
        # used mainly during testing
        self.compile()

    #------------------------------------------------------------------------------------------------------------------
    # Network Configuration
    #------------------------------------------------------------------------------------------------------------------

    def last_layer(self):
        # returns last value from neural network OrderedDict
        # without destroying it
        # used for connecting next neuron
        return list(self.network.values())[-1]


    def make_layer(self, name, input, units, filter_size=(3,3), stride=(1,1), pad=(1,1), alpha=0.25):
        '''
            name is name of layer
        '''
        # Conv2DLayer accepts input layer feeding into this layer
        # units is generator_filter, number of learnable convolutional filters
        # filter_size is size of filter
        conv = ConvLayer(input, units, filter_size, stride=stride, pad=pad, nonlinearity=None)
        # param rectify rectifies nonlinearity
        # import for neural network image classification
        # from Delving Deep into Rectifiers (Kaiming He, 2015)
        prelu = lasagne.layers.ParametricRectifierLayer(conv, alpha=lasagne.init.Constant(alpha))
        # add layer to neural network OrderedDict
        self.network[name+'x'] = conv
        # add layer to neural network OrderedDict
        self.network[name+'>'] = prelu
        # return the parametric rectifier
        return prelu


    def make_block(self, name, input, units):
        # create another layer
        self.make_layer(name+'-A', input, units, alpha=0.1)
        # performs elementwise sum of input layers,
        # which all must have same shape
        return ElemwiseSumLayer([input, self.last_layer()]) if args.generator_residual else self.last_layer()


    def setup_generator(self, input, config):
        # iter dict of string keys corresponding to user input
        # including residual, downscale, upscale, filters
        # and values associated with them
        for k, v in config.items():
            # add attributes to python args values
            setattr(args, k, v)
        # raises diff of image scale to exponent of 2
        # each side factor x increases area by 2**x
        args.zoom = 2**(args.generator_upscale - args.generator_downscale)
        # extend transforms multiple arrays into one array
        units_iter = extend(args.generator_filters)
        # returns next item in iterator
        units = next(units_iter)
        # create input layer of neural network
        self.make_layer('iter.0', input, units, filter_size=(7,7), pad=(3,3))
        # creates hidden layers from downscale generator
        for i in range(0, args.generator_downscale):
            # create downscale layers from last layer and next generator_filters
            self.make_layer('downscale%i'%i, self.last_layer(), next(units_iter), filter_size=(4,4), stride=(2,2))

        # get next iterator
        units = next(units_iter)
        # iterate through generator_blocks
        for i in range(0, args.generator_blocks):
            # create new block which sums layers
            self.make_block('iter.%i'%(i+1), self.last_layer(), units)

        # creates hidden layers for upscale generator neural networks
        for i in range(0, args.generator_upscale):
            # next fitler
            u = next(units_iter)
            # create new upscale layer for neural network
            # u*4 is larger filter size
            self.make_layer('upscale%i.2'%i, self.last_layer(), u*4)
            # add new layer to neural network
            # generate SubpixelReshuffleLayer object
            # at specific dict in neural network
            self.network['upscale%i.1'%i] = SubpixelReshuffleLayer(self.last_layer(), u, 2)
        # create output neuron in network using lasagne Conv2DLayer
        self.network['out'] = ConvLayer(self.last_layer(), 3, filter_size=(7,7), pad=(3,3), nonlinearity=None)


    def list_generator_layers(self):
        # lasagne get_all_layers collects all layers just below the output layer
        # layers are preceded by layers they are dependent upon
        # network['out'] is neural network output, network['img'] is input
        # yeild keyword returns a generator for name and lasagne layer
        for l in lasagne.layers.get_all_layers(self.network['out'], treat_as_input=[self.network['img']]):
            # skip if not params
            if not l.get_params(): continue
            # make a dict of keys, index of layer
            name = list(self.network.keys())[list(self.network.values()).index(l)]
            # return generator of name, lasagne layer pair
            yield (name, l)


    # simple function to return filename of neural network
    def get_filename(self, absolute=False):
        # look for pretrained network based on zoo, type, model
        filename = 'ne%ix-%s-%s-%s.pkl.bz2' % (args.zoom, args.type, args.model, __version__)
        # return absoulte path of the network
        return os.path.join(os.path.dirname(__file__), filename) if absolute else filename


    def load_model(self):
        # condition checks if path of neural network exists
        if not os.path.exists(self.get_filename(absolute=True)):
            # return empty dicts if input for training
            if args.train: return {}, {}
            # error message if neural network not found and not training
            error("Model file with pre-trained convolution layers not found. Download it here...",
                  "https://github.com/alexjc/neural-enhance/releases/download/v%s/%s"%(__version__, self.get_filename()))
        # output which model is being used for training
        print('  - Loaded file `{}` with trained model.'.format(self.get_filename()))
        # opens bzip2-compressed file in binary mode, return file object
        # pickle reads object representation from file object file
        return pickle.load(bz2.open(self.get_filename(absolute=True), 'rb'))


    def load_generator(self, params):
        # return if there are no params
        if len(params) == 0: return
        # generator for name and layer of lasagne layer
        for k, l in self.list_generator_layers():
            # error if layer is not found
            assert k in params, "Couldn't find layer `%s` in loaded model.'" % k
            # error if layers are not same size, which is a mismatch
            assert len(l.get_params()) == len(params[k]), "Mismatch in types of layers."
            # itererate over lasagne parameters
            for p, v in zip(l.get_params(), params[k]):
                # determines if shape of numpy array is same,
                # since each layer must be the same size
                assert v.shape == p.get_value().shape, "Mismatch in number of parameters for layer {}.".format(k)
                # p is theano TensorSharedVariable
                # sets shared variable of p to numpy array
                p.set_value(v.astype(np.float32))

    #------------------------------------------------------------------------------------------------------------------
    # Training & Loss Functions
    #------------------------------------------------------------------------------------------------------------------

    def compile(self):
        # Helper function for rendering test images during training, or standalone inference mode.
        # required input tensor vars
        input_tensor, seed_tensor = T.tensor4(), T.tensor4()
        # create dict of input neuron 'img' and next neuron 'seed'
        # and connects value to tensorflow inputs
        input_layers = {self.network['img']: input_tensor, self.network['seed']: seed_tensor}
        # computes output of network at given layers,
        # including seed and out, and is given optional input layer
        # the output is tensorflow objects
        output = lasagne.layers.get_output([self.network[k] for k in ['seed','out']], input_layers, deterministic=True)
        # compiles graph into callable object
        # called a theano.function
        # which is called in the for loop in model.process
        self.predict = theano.function([seed_tensor], output)


# neural enhance class
class NeuralEnhancer(object):
    # constructor predominantly used to output console/error messages
    # and initialize dataloader and model used
    def __init__(self, loader):
        # condition for error if no files input
        if len(args.files) == 0: error("Specify the image(s) to enhance on the command-line.")
        # output to user the images specified
        print('{}Enhancing {} image(s) specified on the command-line.{}'\
              .format(ansi.BLUE_B, len(args.files), ansi.BLUE))

        # init DataLoader if training required, ie loader == true
        # thread == None if no training required
        # self.thread = DataLoader() if loader else None
        self.thread = None
        # init model object
        self.model = Model()
        # resets console colors
        print('{}'.format(ansi.ENDC))

    # wrapper to call scipy image save function
    def imsave(self, fn, img):
        # call scipy image save function
        scipy.misc.toimage(np.transpose(img + 0.5, (1, 2, 0)).clip(0.0, 1.0) * 255.0, cmin=0, cmax=255).save(fn)

    def process(self, original):
        # Snap the image to a shape that's compatible with the generator (2x, 4x)
        # raises image scale to exponent of 2
        # each side factor x increases area by 2**x
        s = 2 ** max(args.generator_upscale, args.generator_downscale)
        # original is image as numpy array
        # by, bx number of rows, cols respectively of numpy image array
        # and not flow over image factor of change
        by, bx = original.shape[0] % s, original.shape[1] % s
        # trims image
        original = original[by-by//2:original.shape[0]-by//2,bx-bx//2:original.shape[1]-bx//2,:]
        # Prepare paded input image as well as output buffer of zoomed size.
        # shorten names of arguments, no need to reference
        s, p, z = args.rendering_tile, args.rendering_overlap, args.zoom
        # pads the numpy array
        image = np.pad(original, ((p, p), (p, p), (0, 0)), mode='reflect')
        # creates empty array of size orig * factor increase
        output = np.zeros((original.shape[0] * z, original.shape[1] * z, 3), dtype=np.float32)
        # Iterate through the tile coordinates and pass them through the network.
        for y, x in itertools.product(range(0, original.shape[0], s), range(0, original.shape[1], s)):
            # creates new array with axes transpose
            img = np.transpose(image[y:y+p*2+s,x:x+p*2+s,:] / 255.0 - 0.5, (2, 0, 1))[np.newaxis].astype(np.float32)
            # tensor predict
            # TODO elaborate on this more
            *_, repro = self.model.predict(img)
            # place predicted image range into output image range
            output[y*z:(y+s)*z,x*z:(x+s)*z,:] = np.transpose(repro[0] + 0.5, (1, 2, 0))[p*z:-p*z,p*z:-p*z,:]
            print('.', end='', flush=True)
        # clip limits values values limited to between 0 and 1,
        # then multiplied by 255 which is max pixel value
        output = output.clip(0.0, 1.0) * 255.0
        # returns an image from numpy array
        return scipy.misc.toimage(output, cmin=0, cmax=255)


# entry point of program
if __name__ == "__main__":
    # initialize enhancer object
    # loader == false if no training
    enhancer = NeuralEnhancer(loader=False)
    # loops through each file passed as args
    for filename in args.files:
        # prints each filename args
        print(filename, end=' ')
        # TODO change this to video file
        # imread reads multiple files in color
        img = scipy.ndimage.imread(filename, mode='RGB')
        # calls the major functionality of the enhancer object
        # passing called image to be processed
        out = enhancer.process(img)
        # saves new image
        out.save(os.path.splitext(filename)[0]+'_ne%ix.png' % args.zoom)
        # flush ensures that output goes to destination
        print(flush=True)
    # clear colors for console output
    print(ansi.ENDC)


# end of file
