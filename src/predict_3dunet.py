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

import unet_models as un

# read labels
rl_data = sio.loadmat('labels.mat')
l_data = rl_data['labels']
labels_data = l_data[0]

# vol_size
vol_size = (160, 192 ,224)

# set model directory
#m_dir = '../models/slice20_30.h5'

# set model
#model = un.unet(pretrained_weights = m_dir, label_nums = len(labels_data))

# slice the image


# plot the image




