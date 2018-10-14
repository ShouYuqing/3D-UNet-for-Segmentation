# import lib
import os
import glob
import sys
import random
from argparse import ArgumentParser
import scipy.io as sio
import numpy as np

import keras
from keras.models import load_model, Model
import tensorflow as tf
sys.path.append('../ext/neuron')
import neuron.plot as nplt
import unet_models as un

# read labels
rl_data = sio.loadmat('labels.mat')
l_data = rl_data['labels']
labels_data = l_data[0]

# read data
vol_dir='../data/'
atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol']
atlas_seg = atlas['seg']

# slice the image
slice_vol = atlas_vol[100]
slice_seg = atlas_seg[100]
# vol_size
vol_size = (160, 192 ,224)

# set model directory
m_dir = '../models/slice100_30.h5'


# set model
load_model = un.unet(pretrained_weights = m_dir, label_nums = len(labels_data))

# predict
p_outcome = load_model.predict(atlas_vol)


# plot the image




