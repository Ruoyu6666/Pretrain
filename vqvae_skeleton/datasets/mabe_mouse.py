import os
import pickle
import logging
import numpy as np
import torch

from .dataset import SkeletonDataset
from torchvision import transforms



class MabeMouseDataset(SkeletonDataset):
    DEFAULT_GRID_SIZE = 850

    def __init__(self, 
                 path_to_data_dir, 
                 sampling_rate = 1, 
                 num_frames = 80, 
                 sliding_window = 30, 
                 if_fill = True,
                 patch_size: tuple = (6, 1, 2), 
                 cache_path = None, 
                 cache = True, 
                 augmentations: transforms.Compose = None, #centeralign: bool = False, 
                 include_testdata: bool = False,
                 **kwargs):
        
        super().__init__(path_to_data_dir, sampling_rate, num_frames, sliding_window, if_fill, patch_size, 
                         cache_path, cache, **kwargs) # calls Baseclass.__init__(self, ....)
        
        self.augmentations = augmentations
        self.include_testdata = include_testdata
    

    def load_data(self):
        """Load raw data"""
        self.raw_data = np.load(self.path_to_data_dir, allow_pickle=True).item()


    def check_annotations(self) -> None:
        """Annotation check handler"""
        self.has_annotations = "vocabulary" in self.raw_data.keys()
        if self.has_annotations:
            self.annotation_names = self.raw_data["vocabulary"]


    @staticmethod
    def fill_holes(data):
        """Fill zero """
        clean_data = data.copy()
        num_frames, num_individuals, num_joints, _ = clean_data.shape
        # Fill frame 0 using future frames
        for m in range(num_individuals):
            holes = np.where(clean_data[0, m, :, 0] == 0)[0]
            for h in holes:
                valid = np.where(clean_data[:, m, h, 0] != 0)[0]
                if valid.size > 0:
                    clean_data[0, m, h, :] = clean_data[valid[0], m, h, :]
        # Forward-fill remaining frames
        for fr in range(1, num_frames):
            for m in range(num_individuals):
                holes = np.where(clean_data[fr, m, :, 0] == 0)[0]
                clean_data[fr, m, holes, :] = clean_data[fr - 1, m, holes, :]


    def preprocess(self):
        """Initial preprocessing"""
        self.check_annotations()

        sequences = self.raw_data["sequences"]
        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        self.labels = {key: [] for key in self.annotation_names}
        
        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()): #index ,(mouse_name, value)
            vec_seq = sequence["keypoints"]
            if self.if_fill:
                vec_seq = self.fill_holes(vec_seq) # 1800, 3, 12, 2
            if self.sampling_rate > 1:
                vec_seq = vec_seq[:: self.sampling_rate]
            seq_keypoints.append(vec_seq)
            for i in range(len(self.annotation_names)):
                self.labels[self.annotation_names[i]].append(sequence["annotations"][i])
            
            keypoints_ids.extend([(seq_ix, i) for i in np.arange(0, len(vec_seq) - sub_seq_length + 1, self.sliding_window)])
        
        self.seq_keypoints = np.array(seq_keypoints, dtype=np.float32) # (1600 * 30, 900, 3, 12, 2)
        self.keypoints_ids = keypoints_ids
        for label_name in self.annotation_names:
            self.labels[label_name] = np.array(self.labels[label_name], dtype=np.float32)

        del self.raw_data

    def save_processed_data(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, "wb") as output:
            pickle.dump({"keypoints": self.seq_keypoints, "labels": self.labels}, output)
        logging.info("Processed data was saved to {}.".format(self.cache_path))


    def load_from_processed(self):
        logging.warning( f"Loading processed data from {self.cache_path}. "
                         "Delete this file or set cache=False if processing changed.")
        with open(self.cache_path, "rb") as fp:
            self.seq_keypoints, self.labels = pickle.load(fp)

    def normalize(self, data): # for one sample
        #Scale by dimensions of image and mean-shift to center of image.
        state_dim = data.shape[1] // 2
        shift = [int(self.DEFAULT_GRID_SIZE / 2), int(self.DEFAULT_GRID_SIZE / 2),] * state_dim
        scale = [int(self.DEFAULT_GRID_SIZE / 2), int(self.DEFAULT_GRID_SIZE / 2),] * state_dim
        return np.divide(data - shift, scale)

    def prepare_subsequence_sample(self, sequence: np.ndarray): # prepare one sample for __getitem__
        """
        input sequence :(self.max_keypoints_le, 3, 12, 2)
        Returns a training sample
        """
        if self.augmentations:
            sequence = self.augmentations(sequence)
        """Simplest case: Flatten"""
        sequence = sequence.reshape(self.max_keypoints_len, -1)
        keypoints = self.normalize(sequence)
        #if self.centeralign:
        #    keypoints = keypoints.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
        #    keypoints = self.transform_to_centeralign_components(keypoints)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        feats = torch.unsqueeze(keypoints, 0)

        return feats
    
    def __len__(self):
        return len(self.keypoints_ids)
    
    def __getitem__(self, idx: int):
        subseq_ix = self.keypoints_ids[idx]
        subsequence = self.seq_keypoints[subseq_ix[0], subseq_ix[1] : subseq_ix[1] + self.max_keypoints_len] # 900, 3, 12, 2
        feats = self.prepare_subsequence_sample(subsequence)
        
        return feats, [] # feats:[1, 900, 72]
