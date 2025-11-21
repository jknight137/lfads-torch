import logging
import shutil
from glob import glob
from pathlib import Path
import pickle
import h5py
import torch
from tqdm import tqdm
import numpy as np
import pandas as pd
import json
import snel_toolkit.decoding as dec
from snel_toolkit.analysis import PSTH
from snel_toolkit.utils import smooth
import sklearn
from scipy.special import gammaln
from ..datamodules import reshuffle_train_valid
from ..tuples import SessionOutput
from ..utils import send_batch_to_device, transpose_lists

logger = logging.getLogger(__name__)

class NeuralinkEval(pl.Callback):

    def __init__(self)
        

    def on_fit_end(self, trainer, pl_module):

        