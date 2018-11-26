import os
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio
import numpy as np

#import tensorflow and keras
import skimage.io as io
import skimage.transform as trans
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model


# neuron and other libraries
sys.path.append('../ext/neuron')
sys.path.append('../ext/medipy-lib')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
#import losses
import pytools.patchlib as palib
import neuron.generators as genera
import neuron.metrics as metrics
import datagenerators
import unet_models as un


# read model
test_iter =
m_dir = '/home/ys895/unet/iter' + str(test_iter) + '.h5'

# read validation data
valid_file = open('../data/validate_data.txt')
valid_strings = valif_file.readlines()
lenn = 5
vol_list = list() # list of volume data
seg_list = list() # list of segmentation data
for i in range(0,lenn):
    st = train_strings[i].strip()
    vol_dir = '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/' + st
    seg_dir = '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/' + st
    X_vol, X_seg = datagenerators.load_example_by_name(vol_dir, seg_dir)
    vol_list.append(X_vol)
    seg_list.append(X_seg)

# test the data on the model



#