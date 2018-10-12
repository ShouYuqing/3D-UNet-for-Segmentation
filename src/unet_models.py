# main imports
import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Conv3D, Activation, Input, UpSampling3D, concatenate
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import neuron.layers as nrn_layers
import neuron.utils as nrn_utils

# other vm functions
import losses

def myunet(enc_nf, dec_nf, full_size=True, input):
    # inputs
    #src = Input(shape=vol_size + (1,))
    #tgt = Input(shape=vol_size + (1,))
    #x_in = concatenate([src, tgt])

    # down-sample path (encoder)
    x_enc = [input]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = UpSampling3D()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = UpSampling3D()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = UpSampling3D()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])

    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields
    # that are 1/2 size
    if full_size:
        x = UpSampling3D()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])
    print(x.shape)

    #return Model(inputs=[src, tgt], outputs=[x])
