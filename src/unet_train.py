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
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
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
#import my library
#import unet_models as un

# define callback function
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.accuracy = []

    def on_epoch_end(self, batch, logs={}):
        self.accuracy.append(logs.get('dice_coef'))
history = LossHistory()
A = []

# dir
model_name = 'unet1'
m_dir='/home/ys895/unet/' +model_name + '.h5'

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
label_num = len(labels)
# get data patch & training model
#vol_patch = genera.patch(X_vol, patch_size = [64,64,64])
#seg_patch = genera.patch(X_seg, patch_size = [64,64,64])
iter_times = lenn
# define model of unet
#model = un.unet(pretrained_weights = m_dir, label_num=label_num)
model = un.unet(label_num=label_num+1)

for i in range(0, 19):
    #rand_num = random.randint(0, 18)
    ii = i%19
    X_vol = vol_list[ii]
    X_seg = seg_list[ii]
    for vol,arg in palib.patch_gen(X_vol[0, :, :, :, 0], patch_size=[64, 64, 64], stride=32, nargout=0):
    #for vol in genera.patch(X_vol[0, :, :, :, 0], patch_size=[64, 64, 64], patch_stride=32):
        arg_arr = np.array(arg)
        # get segmentation data
        seg=X_seg[0,arg_arr[0], arg_arr[1], arg_arr[2],0]
        seg = genera._relabel(seg, labels=labels)
        #print(seg.shape)
        seg = seg.astype(np.int64)
        seg = genera._categorical_prep(seg, nb_labels_reshape = 31, keep_vol_size = True, patch_size = [64,64,64])
        #seg = metrics._label_to_one_hot(seg, nb_labels=31)
        #print(seg.shape)
        #seg = seg.reshape((1,)+ seg.shape +(1,))
        # adjust data
        vol = np.reshape(vol, (1,) + vol.shape + (1,))
        #seg = np.reshape(seg, (1,) + vol.shape + (1,))
        # relabel segmentation data into  one-hot encoding


        # train
        print('volume ' + str(i) + 'training...')
        #a = model.fit(vol, seg)
        a = model.fit(vol, seg, callbacks=[history])
        #print(a.history)
        #b = float(a.history['dice_coeff'][0])
        #A.append(b)
        # save model
        #if step % model_save_iter == 0:
        #    model.save(os.path.join(model_dir, 'slice' + str(i) + '_' + str(pre_num + step) + '.h5'))
        #step = step + 1

#model.save(os.path.join(model_dir, 'slice' + str(i) + '_' + str(pre_num + step) + '.h5'))
print(A)


        #print(seg.shape)

        #print(arg.shape)
        #print(arg_arr)
        #print(arg_arr.shape)
    #seg = genera.patch(X_seg[0, :, :, :, 0], patch_size=[64, 64, 64], patch_stride=32)
    #print(vol_p.shape)
    #for vol,seg in genera.patch(X_vol[0,:,:,:,0], patch_size = [64,64,64],patch_stride=32), genera.patch(X_seg[0,:,:,:,0], patch_size = [64,64,64],patch_stride=32):
    #   print(vol.size)

