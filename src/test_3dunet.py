import os
import glob
import sys
import random
from argparse import ArgumentParser
import tensorflow as tf
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

def test():

    # read model

    # read data

    # test