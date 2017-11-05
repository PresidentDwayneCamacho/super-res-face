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

# init the version
__version__ = '0.3'

# interact with operating system
# for directory information
import os
# interact with user/terminal window
import sys
# reads .bz2 files
# and creates data stream
import bz2
# pickle allows package of json files
import pickle
# allows arguments to be attached to objects
import argparse
# iterators
import itertools
# collections allows OrderedDict
# which is 'centerpeice' of neural network
import collections


# Configure all options first so we can later custom-load other libraries (Theano) based on device specified by user.
parser = argparse.ArgumentParser(description='Generate a new image by applying style onto a content image.',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# shorten name of parser object/var
add_arg = parser.add_argument
# args to designate files to read
add_arg('files',                nargs='*', default=[])
# args to designate the increase of pixels
add_arg('--zoom',               default=2, type=int,                help='Resolution increase factor for inference.')
# args to designate subdivisions of rendered tiles
add_arg('--rendering-tile',     default=80, type=int,               help='Size of tiles used for rendering images.')
# args to designate padding of pixel tiles
add_arg('--rendering-overlap',  default=24, type=int,               help='Number of pixels padding around each tile.')
# args determines neural network that will be used
add_arg('--type',               default='photo', type=str,          help='Name of the neural network to load/save.')
# args to designate pretrained neural network version
add_arg('--model',              default='default', type=str,        help='Specific trained version of the model.')
# args to designate increase image size
add_arg('--generator-upscale',  default=2, type=int,                help='Steps of 2x up-sampling as post-process.')
# args to designate decrease image size
add_arg('--generator-downscale',default=0, type=int,                help='Steps of 2x down-sampling as preprocess.')
# args to designate number convolution units in network
add_arg('--generator-filters',  default=[64], nargs='+', type=int,  help='Number of convolution units in network.')
# args to designate residual blocks per iteration
add_arg('--generator-blocks',   default=4, type=int,                help='Number of residual blocks per iteration.')
# args to designate layers in block
add_arg('--generator-residual', default=2, type=int,                help='Number of layers in a residual block.')
# args to designate beginning of generator
add_arg('--generator-start',    default=0, type=int,                help='Epoch count to start training generator.')
# args to designate update of descriminator
add_arg('--discriminator-start',default=1, type=int,                help='Epoch count to update the discriminator.')
# args to designate beginning of start
add_arg('--adversarial-start',  default=2, type=int,                help='Epoch for generator to use discriminator.')
# args to designate cpu or gpu
add_arg('--device',             default='cpu', type=str,            help='Name of the CPU/GPU to use, for Theano.')
# args to designate shorten name of parse_args
args = parser.parse_args()

#----------------------------------------------------------------------------------------------------------------------

# Color coded output helps visualize the information
class ansi:
    # yellow for warning
    YELLOW = '\033[0;33m'
    # yellow for warning
    YELLOW_B = '\033[1;33m'
    # red for error
    RED = '\033[0;31m'
    # red for error
    RED_B = '\033[1;31m'
    # clear ansi color info
    ENDC = '\033[0m'

# error message with undefined number of lines to input
def error(message, *lines):
    # generate string with error message
    string = "\n{}ERROR: " + message + "{}\n" + "\n".join(lines) + ("{}\n" if lines else "{}")
    # print error message to user
    print(string.format(ansi.RED_B, ansi.RED, ansi.ENDC))
    # exit from the program if improper neural network
    sys.exit(-1)

# warning message with undefined number of lines
def warn(message, *lines):
    # generate string with warning message
    string = "\n{}WARNING: " + message + "{}\n" + "\n".join(lines) + "{}\n"
    # print warning to the user
    print(string.format(ansi.YELLOW_B, ansi.YELLOW, ansi.ENDC))

# itertools.chain turns multiple arrays into single array
def extend(lst): return itertools.chain(lst, itertools.repeat(lst[-1]))

# Load the underlying deep learning libraries based on the device specified.  If you specify THEANO_FLAGS manually,
# the code assumes you know what you are doing and they are not overriden!
os.environ.setdefault('THEANO_FLAGS', 'floatX=float32,device={},force_device=True,allow_gc=True,'\
                                      'print_active_device=False'.format(args.device))

# scientific & imaging libraries
import numpy as np
# import scipy functionality, and image read function
import scipy.ndimage, scipy.misc, PIL.Image

# numeric computing for gpu
import theano, theano.tensor as T
# set function to neural network softplus
T.nnet.softminus = lambda x: x - T.nnet.softplus(x)

# support ansi colors in windows
if sys.platform == 'win32':
    # colorama library used from ansi color codes
    import colorama

# deep learning framework
import lasagne
# convolution and deconvolution layers from lasagne
from lasagne.layers import Conv2DLayer as ConvLayer, Deconv2DLayer as DeconvLayer, Pool2DLayer as PoolLayer
# beginning layer of neural network
from lasagne.layers import InputLayer, ConcatLayer, ElemwiseSumLayer, batch_norm

#======================================================================================================================
# Convolution Networks
#======================================================================================================================
# reshuffling the neural network
class SubpixelReshuffleLayer(lasagne.layers.Layer):
    '''
        Based on the code by ajbrock: https://github.com/ajbrock/Neural-Photo-Editor/
    '''
    # inherets lasagne.layers.Layer
    def __init__(self, incoming, channels, upscale, **kwargs):
        # inherits from super class
        super(SubpixelReshuffleLayer, self).__init__(incoming, **kwargs)
        # init member variables
        self.upscale = upscale
        # init member variables
        self.channels = channels

    # upscales shape of image if upscale requested
    # returns tuple including channels and upsized shape
    def get_output_shape_for(self, input_shape):
        # upsize the shape of the image if upscale present
        def up(d): return self.upscale * d if d else d
        # return tuple of shape, including channels and upsized shape
        return (input_shape[0], self.channels, up(input_shape[2]), up(input_shape[3]))

    # returns specific subtensor incremented by input
    def get_output_for(self, input, deterministic=False, **kwargs):
        # replicate zero arrays of a given shape
        out, r = T.zeros(self.get_output_shape_for(input.shape)), self.upscale
        # repeat loop once for each int upscale
        for y, x in itertools.product(range(r), repeat=2):
            # return output with given subtensor incremented by input
            out=T.inc_subtensor(out[:,:,y::r,x::r], input[:,r*y+x::r*r,:,:])
        # return the output incremented by upscale
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
            #if args.train: return {}, {}
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
        # output to user the images specified

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


def neural_enhance():
    #args.files = ['img/bruce.jpg']
    # default quantities for arguments
    # no default files
    args.files = []
    # double dimensions of image/video
    args.zoom = 2
    # default model
    args.model = 'default'
    # use photo, which is default
    args.type = 'photo'
    # instanciate NeuralEnhancer object
    enhancer = NeuralEnhancer(loader=False)
    # return instance of NeuralEnhancer
    return enhancer


# end of file
