import argparse
import numpy as np
import pdb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils import *
from models.vqvae import *
from datasets.hbehave_mabe_mouse import MABeMouseDataset



def get_args_parser():

    parser = argparse.ArgumentParser("VQ-VAE Training", add_help=False)

    parser.add_argument( "--job", default="train", const="train", nargs="?", choices=["train", "compute_representations"], help="select task",)
    
    """Model Hyperparameters"""
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_updates", type=int, default=5000)
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
    parser.add_argument("--cache_path", type=str, default='./data/tmp/mabe_mouse_train.pkl')
    parser.add_argument("--cache", default=True, type=str2bool) # if true cache processed data or load from cache

    """Data augmentation and preprocessing"""
    parser.add_argument("--data_augment", default=False, type=str2bool)
    parser.add_argument("--centeralign", action="store_true")
    parser.add_argument("--include_test_data", action="store_true")

    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",)

    """Saving and logging"""
    parser.add_argument("--save_dir", type=str, default="./outputs/") #  models, results, checkpoints
    parser.add_argument("--filename",  type=str, default=timestamp)

    return parser.parse_args()




def train_vqvae(model, loader_train, optimizer, device, writer, timestamp, args):

    if os.path.exists(os.path.join(args.save_dir, 'checkpoints')):
        print('Checkpoint Directory Already Exists - if continue will overwrite files inside. Press c to continue.')
        pdb.set_trace()
    else:
        os.makedirs(os.path.join(args.save_dir, 'checkpoints'))

    model = model.to(device)
    num_epochs = int(args.n_updates / len(loader_train) + 0.5)
    print('Number of epochs to train:', num_epochs)

    results = {'recon_errors': 0, 'total_loss': 0,'perplexities': 0}
    best_loss = float('inf')
    print(len(loader_train))
    for epoch in tqdm(range(1, num_epochs + 1)):
        model.train()
        for i, (x, _)  in enumerate(loader_train):
            x = x.to(device) # [B, 3, 32, 32] for CIFAR10
            optimizer.zero_grad()

            embedding_loss, x_hat, perplexity, _, _ = model(x)
            recon_loss = torch.mean((x_hat - x)**2) #/ x_train_var
            loss = recon_loss + embedding_loss
            loss.backward()
            optimizer.step()

            results["recon_errors"] += recon_loss.item()
            results["perplexities"] += perplexity.item()
            results["total_loss"]   += loss.item()
        
        avg_recon_error = results["recon_errors"] / len(loader_train)
        avg_perplexity = results["perplexities"] / len(loader_train)
        avg_total_loss = results["total_loss"] / len(loader_train)

        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {avg_total_loss:.4f}, '
              f'Recon: {avg_recon_error:.4f}, VQ: {avg_perplexity:.4f}, Perplexity: {avg_perplexity:.2f}')
        
        if avg_total_loss < best_loss:
            best_loss = avg_total_loss
            save_checkpoint(model, optimizer, epoch, args)
            print('Saved best model at epoch ', epoch)
    
    # Save final model and results
    save_model(model, optimizer, args)
    save_results(results, args, timestamp)

    """
    if i % args.log_interval == 0:
        # save model and print value
        if args.save:
            hyperparameters = args.__dict__.save_model_and_results(model, results, hyperparameters, args.filename)
        #writer.add_scalar('Train/Recon_Loss', recon_loss.item(), step)
        #writer.add_scalar('Train/Perplexity', perplexity.item(), step)
        #writer.add_scalar('Train/Total_Loss', loss.item(), step)
    """


# To Check 
def compute_representations(model, loader, args):

    os.makedirs(args.save_dir + '/representations', exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    all_representations = []
    all_encoding =  []
    all_encoding_indices = []

    with torch.no_grad():
        for i, (x, _)  in enumerate(loader_train):
            x = x.to(device)
            z = model.encoder(x)
            z = model.pre_quant_conv(z)
            vq_loss, x_recon, perplexity, min_encodings, min_encodings_indices = model.vq_layer(z)

            all_representations.append(x_recon.cpu().numpy())
            all_encoding.append(min_encodings.cpu().numpy())
            all_encoding_indices.append(min_encodings_indices.cpu().numpy())

    all_representations = np.concatenate(all_representations, axis=0)
    all_encoding = np.concatenate(all_encoding, axis=0)
    all_encoding_indices = np.concatenate(all_encoding_indices, axis=0)

    np.save(args.save_dir + '/representations/vqvae_representations.npy', all_representations)
    #np.save(args.save_dir + '/representations/vqvae_encodings.npy', all_encoding)
    np.save(args.save_dir + '/representations/vqvae_encoding_indices.npy', all_encoding_indices)
    codebook = model.vq_layer.embedding.weight.cpu().numpy()
    np.save(args.save_dir + '/representations/vqvae_codebook.npy', codebook)

    return all_representations




if __name__ == "__main__":

    timestamp = readable_timestamp()

    args = get_args_parser()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    """
    Set up VQ-VAE model with components defined in ./models/ folder
    """
    model = VQVAE(1, args.n_hiddens, args.n_residual_layers,  args.n_residual_hiddens,
                  args.n_embeddings, args.embedding_dim, args.beta).to(device)
    
    """
    Set up data set and data loaders
    """
    #training_data, validation_data, training_loader, validation_loader, x_train_var = load_data_and_data_loaders(args.dataset, args.batch_size)
    dataset_train = MABeMouseDataset(mode="pretrain", path_to_data_dir=args.path_to_data_dir,
                                    num_frames=args.num_frames, sliding_window=args.sliding_window,
                                    sampling_rate=args.sampling_rate, if_fill_holes=args.if_fill_holes,
                                    cache_path=args.cache_path, cache=args.cache,
                                    augmentations=args.data_augment, centeralign=args.centeralign,
                                    include_testdata=args.include_test_data,
                                    #patch_size=args.patch_kernel, q_strides=args.q_strides,
                                    )
    """
    dataset_test = MABeMouseDataset(mode="test", path_to_data_dir=args.path_to_data_dir,
                                    num_frames=args.num_frames, sliding_window=args.sliding_window,
                                    sampling_rate=args.sampling_rate, if_fill_holes=args.if_fill_holes,
                                    augmentations=None, centeralign=args.centeralign,)

    """
    loader_train = DataLoader(dataset_train, #sampler=sampler_train,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.pin_mem, drop_last=True,)
    """
    loader_test = DataLoader(dataset_test, #sampler=sampler_test,
                             batch_size=args.batch_size, num_workers=args.num_workers,
                             pin_memory=args.pin_mem, drop_last=False,)
    """

    """Set up optimizer and training loop"""

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True)
    if args.job == "train":
        train_vqvae(model, loader_train, optimizer, device, None, timestamp, args)
    #elif args.job == "compute_representations":
    #    compute_representations(model, loader_train, args)
