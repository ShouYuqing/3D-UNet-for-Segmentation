import os
import glob
import sys
import random
from argparse import ArgumentParser
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# neuron and other libraries
sys.path.append('../ext/neuron')
sys.path.append('../ext/medipy-lib')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools-lib')
import datagenerators
import losses
import neuron.generators as genera

import unet_models as un
vol_size = (160, 192, 224)


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


def train(model_dir, gpu_id, n_iterations,  model_save_iter, batch_size=1):
    """
    model training
    :param model_dir: the model directory to save to
    :param gpu_id: integer specifying the gpu to use
    :param n_iterations: number of training iterations
    :param model_save_iter: frequency with which to save models
    :param batch_size: Optional, default of 1. can be larger, depends on GPU memory and volume size
    """
    #read label file
    rl_data = sio.loadmat('labels.mat')
    l_data=rl_data['data']
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

    # prepare the model
    #model = un.unet(input_size=(), label_nums =30)

    # if you'd like to initialize the data, you can do it here:
    # model.load_weights(os.path.join(model_dir, '120000.h5'))

    # prepare data for training



    # train
    #for step in range(0, n_iterations):
    for i in range(0, vol_size[1]):
        # set model
        model = un.unet(input_size=vol_size, label_nums=30)

        for (vol_data, seg_data) in genera.vol_seg(vol_data_dir, seg_data_dir,relabel=labels_data,  nb_labels_reshape=len(labels_data),
                                                   iteration_time=n_iterations):
            # get data and adjust data
            vol_train = vol_data[:, :, i, :]
            vol_train = vol_train.reshape(vol_train.shape + (1,))
            seg_train = seg_data[:, :, i, :, :]

            # train
            print('volume ' + str(i) + 'training...')
            model.fit(vol_train, seg_train)
            #train_loss = model.train_on_batch([X, atlas_vol], [atlas_vol, zero_flow])
            #if not isinstance(train_loss, list):
            #    train_loss = [train_loss]

            # print the loss.
            #print_loss(step, 1, train_loss)

            # save model
            if step % model_save_iter == 0:
                model.save(os.path.join(model_dir, 'slice' + str(i) + '_' + str(step) + '.h5'))

# main function
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--iters", type=int,
                        dest="n_iterations", default=150000,
                        help="number of iterations")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=100,
                        help="frequency of model saves")
    parser.add_argument("--model_dir", type=str,
                        dest="model_dir", default='../models/',
                        help="models folder")

    args = parser.parse_args()
    train(**vars(args))