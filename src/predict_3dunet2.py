'''
predict the model which was trained for all slices of images
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

# data direction
vol_data_dir='/home/ys895/resize256/resize256-crop_x32/train/vols/'
seg_data_dir='/home/ys895/resize256/resize256-crop_x32/train/asegs/'

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

# save the slice of raw segmentation
datanew = 'raw_seg.mat'
sio.savemat(datanew, {'raw_seg': slice_seg})


# vol_size
vol_size = (160, 192 ,224)
vol_size1 = (160, 224, 1)

# set model directory
m_dir = '/home/ys895/Models/iter500.h5'

# set model
load_model = un.unet(pretrained_weights = m_dir, input_size=vol_size1, label_nums = len(labels_data))

# predict
p_outcome = load_model.predict(slice_vol)

# visualization of the outputs
print(p_outcome)
datanew = 'output_data.mat'
sio.savemat(datanew, {'output': p_outcome})


# change the dimension of the output


# plot the image
#nplt.slices(p_outcome, show = None)
i=100
# set model
for (vol_data, seg_data) in genera.vol_seg(vol_data_dir, seg_data_dir,relabel=labels_data,  nb_labels_reshape=len(labels_data),
                                                   iteration_time=20):

    # get data and adjust data
    vol_test = vol_data[:, :, i, :, :]
    seg_test = seg_data[:, :, i, :, :]

    #seg_train1 = seg_train[0,:,:,:]
    #datanew = 'seg_data.mat'
    #sio.savemat(datanew, {'seg': seg_train1})
    score = load_model.evaluate(vol_test, seg_test, verbose=0)
    print('Test accuracy:', score[1])


