'''
train several models for different layers
'''

# basic import
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


vol_size = (160, 192, 224)
new_vol_size = (vol_size[0], vol_size[2], 1)

#set the data directory
vol_data_dir='/home/ys895/resize256/resize256-crop_x32/train/vols/'
seg_data_dir='/home/ys895/resize256/resize256-crop_x32/train/asegs/'



#for (vol_data, seg_data) in genera.vol_seg(vol_data_dir,seg_data_dir,nb_labels_reshape =500,iteration_time=2):


    #print('the shape of a:')
    #print(a.shape)
    #print('the shape of b:')
    #print(b.shape)
    #outtt=un.myunet(enc_nf=nf_enc,dec_nf=nf_dec,input=a)
    #print(outtt.shape)
    #random.shuffle(train_vol_names)

"""
In order to keep satisfied model's save name, the number of iteration should be the same as the 
number of checkpoint_iteration.
"""

def train(model_dir, gpu_id, n_iterations,  model_save_iter, pre_num):
    """
    model training
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param n_iterations: number of training iterations
    :param model_save_iter: frequency with which to save models
    """
    #read label file
    rl_data = sio.loadmat('../data/labels.mat')
    l_data=rl_data['labels']
    labels_data=l_data[0]

    # prepare model folder
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    # GPU handling
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # train
    # get every slice's model
    for i in range(90,95):
    #for i in range(1, vol_size[1]):
        if(pre_num != 0):
            # generate model directory
            m_dir='/home/ys895/Models/slice' + str(i) + '_' + str(pre_num) + '.h5'
        else:
            m_dir=None
        # set model
        model = un.unet(pretrained_weights = m_dir, input_size=new_vol_size, label_nums=30)
        step = 1
        for (vol_data, seg_data) in genera.vol_seg(vol_data_dir, seg_data_dir,relabel=labels_data,  nb_labels_reshape=len(labels_data),
                                                   iteration_time=n_iterations):

            # get data and adjust data
            vol_train = vol_data[:, :, i, :, :]
            seg_train = seg_data[:, :, i, :, :]

            #seg_train1 = seg_train[0,:,:,:]
            #datanew = 'seg_data.mat'
            #sio.savemat(datanew, {'seg': seg_train1})

            # train
            print('volume ' + str(i) + 'training...')
            model.fit(vol_train, seg_train, batch_size=30)

            # save model
            if step % model_save_iter == 0:
                model.save(os.path.join(model_dir, 'slice' + str(i) + '_' + str(pre_num + step) + '.h5'))
            step = step + 1

# main function
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--iters", type=int,
                        dest="n_iterations", default=15000,
                        help="number of iterations")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=100,
                        help="frequency of model saves")
    parser.add_argument("--pre", type=int,
                        dest="pre_num", default=0,
                        help="previous training iteration")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='/home/ys895/Models/',
                        help="models folder")

    args = parser.parse_args()
    train(**vars(args))