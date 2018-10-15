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

# read labels
rl_data = sio.loadmat('../data/labels.mat')
l_data = rl_data['labels']
labels_data = l_data[0]

# read data
vol_dir='../data/'
atlas = np.load('../data/atlas_norm.npz')
atlas_vol = atlas['vol']
atlas_seg = atlas['seg']

# slice the image
slice_vol = atlas_vol[:,100,:]
slice_vol = slice_vol.reshape((1,) + slice_vol.shape + (1,))
slice_seg = atlas_seg[:,100,:]

# vol_size
vol_size = (160, 192 ,224)
vol_size1 = (160, 224, 1)

# set model directory
m_dir = '/home/ys895/Models/slice100_30.h5'

# set model
load_model = un.unet(pretrained_weights = m_dir, input_size=vol_size1, label_nums = len(labels_data))

# predict
p_outcome = load_model.predict(slice_vol)
print(p_outcome)
datanew = 'output_data.mat'
sio.savemat(datanew, {'output': p_outcome})


# change the dimension of the output


# plot the image
#nplt.slices(p_outcome, show = None)



