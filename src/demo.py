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


'''
demo for function dice score
'''


tensor1 = np.array([[[0,1,0],[0,1,0],[0,1,0]],[[0,1,0],[0,1,0],[0,1,0]],[[0,1,0],[0,1,0],[0,1,0]]])
tensor2 = np.array([[[0,1,0],[0,1,0],[0,1,0]],[[0,1,0],[0,1,0],[0,1,1]],[[0,1,0],[0,1,0],[0,1,0]]])

dice_score = nm.Dice(nb_labels = len(labels_data), input_type='prob', dice_type='hard',).dice(seg_data,concatenate_outcome)
#vals, _ = dice(tensor1, tensor2, nargout=2)
print('calcualting dice score:')
print(dice_score)
#print(np.mean(vals), np.std(vals))