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
from medipy.metrics import dice

# read model
label_num = 30
test_iter = 50
m_dir = '/home/ys895/unet/iter' + str(test_iter) + '.h5'
model = un.unet(label_num=label_num)
model.load_weights(m_dir)

# get label
labels = sio.loadmat('../data/labels.mat')['labels'][0]

# read validation data
valid_file = open('../data/validate_data.txt')
valid_strings = valid_file.readlines()
lenn = 5
vol_list = list() # list of volume data
seg_list = list() # list of segmentation data
for i in range(0,lenn):
    st = valid_strings[i].strip()
    vol_dir = '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/vols/' + st
    seg_dir = '/home/ys895/resize256/resize256-crop_x32/FromEugenio_prep/labels/' + st
    X_vol, X_seg = datagenerators.load_example_by_name(vol_dir, seg_dir)
    vol_list.append(X_vol)
    seg_list.append(X_seg)

# test the data on the model
# the size of test data is 5
cnt = 1
for i in range(0, 1):
    # rand_num = random.randint(0, 18)
    ii = i % 19
    X_vol = vol_list[ii]
    X_seg = seg_list[ii]

    sum_dice = 0
    cnt2 = 0

    for vol, arg in palib.patch_gen(X_vol[0, :, :, :, 0], patch_size=[64, 64, 64], stride=64, nargout=0):

        cnt2 = cnt2 + 1

        arg_arr = np.array(arg)
        # get segmentation data
        seg = X_seg[0, arg_arr[0], arg_arr[1], arg_arr[2], 0]
        seg = genera._relabel(seg, labels=labels)
        seg = seg.astype(np.int64)
        # make the segmentation data into one-hot in order to test it in the model
        seg = genera._categorical_prep(seg, nb_labels_reshape=30, keep_vol_size=True, patch_size=[64, 64, 64])
        # seg = metrics._label_to_one_hot(seg, nb_labels=31)
        # print(seg.shape)
        # seg = seg.reshape((1,)+ seg.shape +(1,))
        # adjust data
        vol = np.reshape(vol, (1,) + vol.shape + (1,))
        pred = model.predict(vol)
        dice_score = Dice(nb_labels = 30,
                 weights=None,
                 input_type='prob',
                 dice_type='hard',
                 approx_hard_max=True,
                 vox_weights=None,
                 crop_indices=None,
                 area_reg=0.1).dice(seg,pred)
        sum_dice = sum_dice + dice_score
        #vals, _ = dice(pred, seg, nargout=2)
        #sum_dice = sum_dice + np.mean(vals)
        #print(np.mean(vals), np.std(vals))
#
print(sum_dice/cnt2)