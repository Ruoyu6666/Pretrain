import os
import pickle
import logging
from pathlib import Path

from abc import ABC, abstractmethod
from torch.utils.data import Dataset


class SkeletonDataset(Dataset, ABC):
    def __init__(self, 
                 path_to_data_dir: Path,
                 sampling_rate: int = 1,  #num_frames: int = 80,
                 sliding_window: int = 1, if_fill: bool = True,
                 patch_size: tuple = (4, 1, 2),
                 cache_path: Path = None, cache: bool = True,
                  **kwargs):

        self.path_to_data_dir = path_to_data_dir
        self.sampling_rate = sampling_rate
        #self.num_frames = num_frames
        self.sliding_window = sliding_window
        self.if_fill = if_fill
        self.patch_size = patch_size

        self.cache_path = cache_path
        if (not cache) or (cache_path is None) or not (os.path.exists(cache_path)):
            self.load_data()    # load raw  data
            self.preprocess()   # initial preprocess
            if cache and cache_path is not None:
                self.save_processed_data()
        else:
            self.load_from_processed() 

        self.max_keypoints_len = self.seq_keypoints.shape[1]
        
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self):
        pass
    
    @abstractmethod
    def save_processed_data(self):
        pass

    @abstractmethod
    def load_from_processed(self):
        pass

    # To do
    def __len__(self):
        return len(self.keypoints_ids)
    # To do
    def __getitem__(self, idx):
        return self.data[idx]