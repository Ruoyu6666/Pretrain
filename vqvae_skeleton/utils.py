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



def save_checkpoint(model, optimizer, epoch, args):
    SAVE_CHECKPOINT_PATH = args.save_dir + '/checkpoints'
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
        'args': args
    }
    torch.save(checkpoint, SAVE_CHECKPOINT_PATH + '/vqvae_checkpoint_epoch_' + str(epoch) + '.pth')



def save_model(model, optimizer, args):
    SAVE_MODEL_PATH = args.save_dir + '/models'
    model = {
        'model': model.state_dict(), 
        'optimizer': optimizer.state_dict(),
        'args': args
        }
    torch.save(model, SAVE_MODEL_PATH + '/vqvae_model'+'.pth')    



def load_checkpoint(model, optimizer, filename, device):
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    args = checkpoint['args']
    return model, optimizer, epoch, args


def save_results(results, args, timestamp):
    SAVE_RESULT_PATH = args.save_dir + '/results'
    torch.save(results, SAVE_RESULT_PATH + '/vqvae_results_' + timestamp + '.pth');
