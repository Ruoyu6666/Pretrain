import os
import time
import numpy as np
import argparse

import torch
from torch.utils.data import DataLoader


def readable_timestamp():
    return time.ctime().replace('  ', ' ').replace(' ', '_').replace(':', '_').lower()



def str2bool(v):
    if type(v) == bool:
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def save_model_and_results(model, results, hyperparameters, timestamp):
    SAVE_MODEL_PATH = os.getcwd() + '/results'
    results_to_save = {
        'model': model.state_dict(),
        'results': results,
        'hyperparameters': hyperparameters
    }
    torch.save(results_to_save,SAVE_MODEL_PATH + '/vqvae_data_' + timestamp + '.pth')


