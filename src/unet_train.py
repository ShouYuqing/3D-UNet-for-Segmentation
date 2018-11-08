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
import losses
import neuron.generators as genera

#import my library
import unet_models as un

# read train data
train_file = open('../data/train_data.txt')
train_strings = train_file.readlines()
lenn = 19
vol_list = list() # list of volume data
seg_list = list() # list of segmentation data
for i in range(0,lenn):
    st = train_strings[i]
    #train_add = np.load(st.strip())
    X_vol, X_seg = datagenerators.load_example_by_name(
        '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/' + st.strip(),
        '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/' + st.strip())
    #train_add = np.reshape(train_add,(1,)+train_add.shape+(1,))
    vol_list.append(X_vol)
    seg_list.append(X_seg)

# get data patch


