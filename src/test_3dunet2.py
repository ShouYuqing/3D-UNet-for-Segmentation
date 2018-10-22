'''
test the model which was trained for all slices of images
'''

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
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/medipy-lib')
import neuron.plot as nplt
import unet_models as un
import neuron.generators as genera

# number of iteration for testing
test_num = 1000

# data direction
vol_data_dir='/home/ys895/resize256/resize256-crop_x32/train/vols/'
seg_data_dir='/home/ys895/resize256/resize256-crop_x32/train/asegs/'

# read labels
rl_data = sio.loadmat('../data/labels.mat')
l_data = rl_data['labels']
labels_data = l_data[0]

# construct the 3D images

# set model directory
m_dir = '/home/ys895/Models/iter' + test_num + '.h5'

# load weight
load_model = un.unet(pretrained_weights = m_dir, input_size=vol_size1, label_nums = len(labels_data))

# test the model with 3D images
for i in range(0,1000):
    for (vol_data,seg_data) in genera.seg_vol()
