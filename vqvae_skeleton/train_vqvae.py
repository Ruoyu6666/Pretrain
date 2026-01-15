import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from models.vqvae import *
from datasets.mabe_mouse import MABeMouseDataset

timestamp = readable_timestamp()
parser = argparse.ArgumentParser()

"""Model Hyperparameters"""
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_updates", type=int, default=5000)
parser.add_argument("--n_hiddens", type=int, default=128)         # h_dim
parser.add_argument("--n_residual_hiddens", type=int, default=32) # res_h_dim
parser.add_argument("--n_residual_layers", type=int, default=2)   # n_res_layers
parser.add_argument("--embedding_dim", type=int, default=64)      # e_dim
parser.add_argument("--n_embeddings", type=int, default=512)      # K: n_e
parser.add_argument("--beta", type=float, default=.25)
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--log_interval", type=int, default=50)


"""Dataset and DataLoader parameters"""
parser.add_argument("--dataset",  type=str, default='mabe_mouse')
parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/data/MaBe/mouse/mouse_triplet_train.npy')
parser.add_argument("--sliding_window", default=1, type=int)
parser.add_argument("--if_fill_holes", default=False, type=str2bool)
parser.add_argument("--cache_path", type=str, default='./cache/mabe_mouse_vqvae_cache.npy')
parser.add_argument("--cache", default=True, type=str2bool)
parser.add_argument("--data_augment", default=False, type=str2bool)
parser.add_argument("--centeralign", action="store_true")
parser.add_argument("--include_test_data", action="store_true")

# whether to save model
parser.add_argument("--save", action="store_true")
parser.add_argument("--filename",  type=str, default=timestamp)

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if args.save:
    print('Results will be saved in ./results/vqvae_' + args.filename + '.pth')


"""
Load data and define batch data loaders
"""
#training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(args.dataset, args.batch_size)

# dataset
dataset_train = MABeMouseDataset(mode="pretrain", path_to_data_dir=args.path_to_data_dir,
                                num_frames=args.num_frames, sliding_window=args.sliding_window,
                                sampling_rate=args.sampling_rate, if_fill_holes=args.if_fill_holes,
                                cache_path=args.cache_path, cache=args.cache,
                                augmentations=args.data_augment, centeralign=args.centeralign,
                                include_testdata=args.include_test_data,
                                #patch_size=args.patch_kernel, q_strides=args.q_strides,
                                )


dataset_test = MABeMouseDataset(mode="test", path_to_data_dir=args.path_to_data_dir,
                                num_frames=args.num_frames, sliding_window=args.sliding_window,
                                sampling_rate=args.sampling_rate, if_fill_holes=args.if_fill_holes,
                                augmentations=None, centeralign=args.centeralign,)
#################################################################################################
# Data Loader
loader_train = DataLoader(dataset_train, #sampler=sampler_train,
                          batch_size=args.batch_size, num_workers=args.num_workers,
                          pin_memory=args.pin_mem, drop_last=True,)

loader_test = DataLoader(dataset_test, #sampler=sampler_test,
                         batch_size=args.batch_size, num_workers=args.num_workers,
                         pin_memory=args.pin_mem, drop_last=False,)

"""
Set up VQ-VAE model with components defined in ./models/ folder
"""
model = VQVAE(args.n_hiddens, args.n_residual_hiddens,args.n_residual_layers, 
              args.n_embeddings, args.embedding_dim, args.beta).to(device)

"""Set up optimizer and training loop"""
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
model.train()

results = {
    'n_updates': 0, 
    'recon_errors': [],
    'loss_vals': [], 
    'perplexities': [],
}

def train():

    for i in range(args.n_updates):
        (x, _) = next(iter(loader_train))
        x = x.to(device) # [32, 3, 32, 32] batch size 32: 
        optimizer.zero_grad()

        embedding_loss, x_hat, perplexity = model(x)
        recon_loss = torch.mean((x_hat - x)**2) / x_train_var
        loss = recon_loss + embedding_loss
        loss.backward()
        optimizer.step()

        results["recon_errors"].append(recon_loss.cpu().detach().numpy())
        results["perplexities"].append(perplexity.cpu().detach().numpy())
        results["loss_vals"].append(loss.cpu().detach().numpy())
        results["n_updates"] = i

        if i % args.log_interval == 0:
            """save model and print values"""
            if args.save:
                hyperparameters = args.__dict__.save_model_and_results(model, results, hyperparameters, args.filename)

            print('Update #', i, 'Recon Error:',
                  np.mean(results["recon_errors"][-args.log_interval:]),
                  'Loss', np.mean(results["loss_vals"][-args.log_interval:]),
                  'Perplexity:', np.mean(results["perplexities"][-args.log_interval:]))


if __name__ == "__main__":
    train()






# Dataset
