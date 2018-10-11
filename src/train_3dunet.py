sys.path.append('../ext/neuron')

import os
import glob
import sys
import random
from argparse import ArgumentParser

# third-party imports
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# project imports
import datagenerators
import networks
import losses


vol_size = (160, 192, 224)

base_data_dir = '/home/ys895/resize256/resize256-crop_x32/'
train_vol_names = glob.glob(base_data_dir + 'train/vols/*.npz')
train_seg_names = glob.glob(base_data_dir + 'train/asegs/*.npz')
random.shuffle(train_vol_names)
