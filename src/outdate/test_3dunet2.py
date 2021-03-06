'''
test the model which was trained for all slices of images by dice scores
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
import neuron.metrics as nm
import neuron.plot as nplt
import unet_models as un
import neuron.generators as genera
from medipy.metrics import dice

# number of iteration for testing
iter_num = 20
test_iter = 8000
# data direction
vol_data_dir='/home/ys895/resize256/resize256-crop_x32/train/vols/'
seg_data_dir='/home/ys895/resize256/resize256-crop_x32/train/asegs/'

# read labels
rl_data = sio.loadmat('../data/labels.mat')
l_data = rl_data['labels']
labels_data = l_data[0]

# set model directory
m_dir = '/home/ys895/Models/iter' + str(test_iter) + '.h5'

# vol_size
vol_size = (160, 192 ,224)
vol_size1 = (160, 224, 1)

# load weight
load_model = un.unet(pretrained_weights = m_dir, input_size=vol_size1, label_nums = len(labels_data))

# test the model with 3D images
for (vol_data,seg_data) in genera.vol_seg(vol_data_dir, seg_data_dir,relabel=labels_data,  nb_labels_reshape=len(labels_data),
                                                   iteration_time=iter_num):
    concatenate_outcome = np.empty(seg_data.shape)
    for i in range(0,191):
        vol_train = vol_data[:, :, i, :, :]

        # concatenate slices
        #slice_outcome = load_model.predict(slice_vol)
        concatenate_outcome[:,:,i,:,:] = load_model.predict(vol_train)
        #np.concatenate([concatenate_outcome,concatenate_outcome])

    concatenate_outcome.reshape([1,concatenate_outcome.shape[1],concatenate_outcome.shape[2],concatenate_outcome.shape[3],concatenate_outcome.shape[4]])
    #print('the shape of the output:')
    #print(concatenate_outcome.shape)
    # compute the dice score of test example
    print('the dice score of the test is:')
    #dice_score = nm.Dice(nb_labels = len(labels_data), input_type='prob', dice_type='hard',).dice(seg_data,concatenate_outcome)
    #dice_score = dice(concatenate_outcome,,)
    vals, _ = dice(concatenate_outcome, seg_data, nargout=2)
    print(np.mean(vals), np.std(vals))
    #print(dice_score)



"""
2d unet changed for class project
output in (1,256,256,label_nums)
input in (1,x,y,1) 4d tensor
"""

'''
we need to change the output into (1, 256, 256, label_nums)
'''

