import argparse
import numpy as np
import pdb
from tqdm import tqdm
from itertools import islice
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from models.vqvae import *
from datasets import mabe_mouse as mice


def get_args_parser():

    parser = argparse.ArgumentParser("VQ-VAE Compute_Representations", add_help=False)
    
    """Model Hyperparameters"""
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=50000)
    parser.add_argument("--n_hiddens", type=int, default=128)         # h_dim
    parser.add_argument("--n_residual_hiddens", type=int, default=32) # res_h_dim
    parser.add_argument("--n_residual_layers", type=int, default=2)   # n_res_layers
    parser.add_argument("--embedding_dim", type=int, default=64)      # e_dim
    parser.add_argument("--n_embeddings", type=int, default=128)      # K: n_e 512
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--log_interval", type=int, default=50)

    """Dataset and DataLoader parameters"""
    parser.add_argument("--dataset",  type=str, default='mabe_mouse')
    parser.add_argument("--path_to_data_dir", type=str, default='/home/rguo_hpc/myfolder/data/MaBe/mouse/mouse_triplet_train.npy')
    parser.add_argument("--num_frames", default=900, type=int)
    parser.add_argument("--sliding_window", default=1, type=int)
    parser.add_argument("--sampling_rate", default=1, type=int)
    parser.add_argument("--if_fill_holes", default=False, type=str2bool)
    parser.add_argument("--patch_size", default=(3,1,24), type = int )
    parser.add_argument("--cache_path", type=str, default='./data/tmp/mabe_mouse_train.pkl')
    parser.add_argument("--cache", default=False, type=str2bool) # if true cache processed data or load from cache
    parser.add_argument("--compression_factor", type=int, default=24)

    """Data augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=False, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")
    parser.add_argument("--include_test_data", action="store_true")
    """"""
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)

    """Saving and logging"""
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--ckpt_path", type=str, default="./outputs/models/vqvae_model.pth")

    return parser.parse_args()



#def load_data



def compute_representations(model, loader, device ,args):

    os.makedirs(args.save_dir + '/representations', exist_ok=True)


    model = model.to(device)
    model.eval()
    all_representations = []
    all_encoding =  []
    all_encoding_indices = []

    with torch.no_grad():
        for i, (x, _)  in enumerate(loader):
        #for i, (x, _) in enumerate(tqdm(islice(loader, 100), total=100)):
            x = x.to(device)
            z = model.encoder(x)
            z = model.pre_quant_conv(z)
            vq_loss, x_recon, perplexity, min_encodings, min_encodings_indices = model.vq_layer(z)

            all_representations.append(x_recon.cpu().numpy())
            #all_encoding.append(min_encodings.cpu().numpy())
            all_encoding_indices.append(min_encodings_indices.cpu().numpy())

    all_representations = np.concatenate(all_representations, axis=0)
    #all_encoding = np.concatenate(all_encoding, axis=0)
    all_encoding_indices = np.concatenate(all_encoding_indices, axis=0)

    np.save(args.save_dir + '/representations/vqvae_representations.npy', all_representations)
    #np.save(args.save_dir + '/representations/vqvae_encodings.npy', all_encoding)
    np.save(args.save_dir + '/representations/vqvae_encoding_indices.npy', all_encoding_indices)
    codebook = model.vq_layer.embedding.weight.cpu().detach().numpy()
    np.save(args.save_dir + '/representations/vqvae_codebook.npy', codebook)

    return all_representations



if __name__ == "__main__":

    timestamp = readable_timestamp()

    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """
    Load Model
    """
    model = VQVAE(1, args.n_hiddens, args.n_residual_layers,  args.n_residual_hiddens,
                  args.n_embeddings, args.embedding_dim, args.beta, args.compression_factor).to(device)
    model.load_state_dict(torch.load(args.ckpt_path, map_location=device, weights_only=False)["model"])



    dataset_test = mice.MabeMouseDataset(path_to_data_dir=args.path_to_data_dir,
                                            sampling_rate=args.sampling_rate,
                                            num_frames=args.num_frames, 
                                            sliding_window=args.sliding_window,
                                            if_fill=args.if_fill_holes,
                                            patch_size=args.patch_size,
                                            cache_path=args.cache_path, cache=args.cache,
                                            augmentations=None,)

    loader_test = DataLoader(dataset_test, #sampler=sampler_test, batch_size=args.batch_size, 
                             num_workers=args.num_workers, pin_memory=args.pin_mem, drop_last=False,)
    compute_representations(model, loader_test, device, args)
    
    