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
import neuron.generators as genera
import datagenerators
#import my library
#import unet_models as un

# read train data
train_file = open('../data/train_data.txt')
train_strings = train_file.readlines()
lenn = 19
vol_list = list() # list of volume data
seg_list = list() # list of segmentation data
for i in range(0,lenn):
    st = train_strings[i].strip()
    vol_dir = '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/' + st
    seg_dir = '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/' + st
    #print(vol_dir)
    #train_add = np.load(st.strip())
    X_vol, X_seg = datagenerators.load_example_by_name(vol_dir, seg_dir)
    #train_add = np.reshape(train_add,(1,)+train_add.shape+(1,))
    vol_list.append(X_vol)
    seg_list.append(X_seg)

# get label
labels = sio.loadmat('../data/labels.mat')['labels'][0]

# get data patch & training model
#vol_patch = genera.patch(X_vol, patch_size = [64,64,64])
#seg_patch = genera.patch(X_seg, patch_size = [64,64,64])
iter_times = lenn
#for i in range(0, iter_times):
for i in range(0, 1):
    rand_num = random.randint(0, 18)
    X_vol = vol_list[rand_num]
    X_seg = seg_list[rand_num]
    #for i in genera.patch(X_vol[0, :, :, :, 0], patch_size=[64, 64, 64], patch_stride=32):
    #    print(i.shape)
    #seg_p = genera.patch(X_seg[0, :, :, :, 0], patch_size=[64, 64, 64], patch_stride=32)
    #print(vol_p.shape)
    for vol in [genera.patch(X_vol[0,:,:,:,0], patch_size = [64,64,64],patch_stride=32), genera.patch(X_seg[0,:,:,:,0], patch_size = [64,64,64],patch_stride=32)]:
       print(vol.size)

