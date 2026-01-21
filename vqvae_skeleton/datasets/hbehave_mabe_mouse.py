import __future__
import os
from pathlib import Path
import numpy as np
import logging
import pickle

import torch
from torchvision import transforms

from .augmentations import GaussianNoise, Reflect, Rotation
from .hbehave_pose_traj import BasePoseTrajDataset


class MABeMouseDataset(BasePoseTrajDataset):
    """
    Primary Mouse (+Features) dataset.
    """
    DEFAULT_FRAME_RATE = 30
    DEFAULT_GRID_SIZE = 850
    NUM_INDIVIDUALS = 3
    NUM_KEYPOINTS = 12
    KPTS_DIMENSIONS = 2
    KEYFRAME_SHAPE = (NUM_INDIVIDUALS, NUM_KEYPOINTS, KPTS_DIMENSIONS) #3, 12, 2
    DEFAULT_NUM_TRAINING_POINTS = 1600
    DEFAULT_NUM_TESTING_POINTS = 3736
    SAMPLE_LEN = 1800
    NUM_TASKS = 13

    NOSE = "nose"
    EAR_LEFT = "ear_left"
    EAR_RIGHT = "ear_right"
    NECK = "neck"
    FOREPAW_LEFT = "forepaw_left"
    FOREPAW_RIGHT = "forepaw_right"
    CENTER = "center"
    HINDPAW_LEFT = "hindpaw_left"
    HINDPAW_RIGHT = "hindpaw_right"
    TAIL_BASE = "tail_base"
    TAIL_MIDDLE = "tail_middle"
    TAIL_TIP = "tail_tip"

    STR_BODY_PARTS = [
        NOSE, EAR_LEFT, EAR_RIGHT, 
        NECK, FOREPAW_LEFT, FOREPAW_RIGHT,
        CENTER, HINDPAW_LEFT, HINDPAW_RIGHT, 
        TAIL_BASE, TAIL_MIDDLE, TAIL_TIP,
    ]
    BODY_PART_2_INDEX = {w: i for i, w in enumerate(STR_BODY_PARTS)}

    def __init__(self, 
                 mode: str, path_to_data_dir: Path, 
                 scale: bool=True, sampling_rate: int = 1, num_frames: int = 80,
                 sliding_window: int = 1, if_fill_holes: bool = False,
                 patch_size: tuple = (4, 1, 2),
                 cache_path: Path = None, cache=True,
                 augmentations: transforms.Compose = None, centeralign: bool = False, 
                 include_testdata: bool = False,
                 **kwargs,):    # child class args

        super().__init__(path_to_data_dir, scale, sampling_rate, num_frames, 
                         sliding_window, if_fill_holes, **kwargs) # base class

        self.sample_frequency = self.DEFAULT_FRAME_RATE  # downsample frames if needed
        self.mode = mode
        self.centeralign = centeralign
        
        if augmentations:
            gs = (self.DEFAULT_GRID_SIZE, self.DEFAULT_GRID_SIZE)
            self.augmentations = transforms.Compose([Rotation(grid_size=gs, p=0.5), GaussianNoise(p=0.5), Reflect(grid_size=gs, p=0.5),])
        else:
            self.augmentations = None
        """
        self.cache_path = cache_path
        if not os.path.exists(self.cache_path) or not cache:
            self.load_data(include_testdata)
            self.preprocess()
            if cache:
                self.save_processed_data(self.seq_keypoints, self.labels)
        else:
            self.load_from_processed()
        """

    def load_data(self, include_testdata) -> None:
        if self.mode == "pretrain":
            self.raw_data = np.load(self.path, allow_pickle=True).item()
            if include_testdata:
                raw_data_test = np.load(self.path.replace("train", "test"), allow_pickle=True).item()
                self.raw_data["sequences"].update(raw_data_test["sequences"])
                # dict.update(other_dict) adds key–value pairs from other_dict into the dictionary, modifying it in place.
        elif self.mode == "test":
            self.raw_data = np.load(self.path.replace("train", "test"), allow_pickle=True).item()
        else:
            raise ValueError("Invalid mode: {}".format(self.mode))


    def load_labeled_data(self) -> None:
        self.raw_data = np.load(os.path.join(self.path, "mouse_triplet_test.npy"), allow_pickle=True).item()
        self.frame_number_map = np.load(os.path.join(self.path, "frame_number_map.npy"), allow_pickle=True).item()
        # frame_number_map, label_array, vocabulary, task_type
        self.labels = np.load(os.path.join(self.path, "mouse_triplets_test_labels.npy"), allow_pickle=True).item()


    def preprocess(self):

        self.check_annotations()

        sequences = self.raw_data["sequences"]
        seq_keypoints = []
        keypoints_ids = []
        sub_seq_length = self.max_keypoints_len
        sliding_window = self.sliding_window
        self.labels = {key: [] for key in self.annotation_names}

        for seq_ix, (seq_name, sequence) in enumerate(sequences.items()): #index(0-1599) ,(mouse_name, value)
            vec_seq = sequence["keypoints"]
            if self.if_fill_holes:
                vec_seq = self.fill_holes(vec_seq)
            #vec_seq = vec_seq.reshape(vec_seq.shape[0], -1)
            vec_seq = vec_seq.reshape(vec_seq.shape[0], 3, -1)
            if self._sampling_rate > 1:
                vec_seq = vec_seq[:: self._sampling_rate]
            """
            # Pads the beginning and end of the sequence with duplicate frames
            if sub_seq_length < 120:
                pad_length = sub_seq_length
            else:
                pad_length = 120
            pad_vec = np.pad(vec_seq,((pad_length // 2, pad_length - 1 - pad_length // 2), (0, 0)), mode="edge", )
            seq_keypoints.append(pad_vec)
            keypoints_ids.extend([(seq_ix, i) for i in np.arange(0, len(pad_vec) - sub_seq_length + 1, sliding_window)])
                                                                    # 1800 + 120 - 900 + 1
            """
            seq_keypoints.append(vec_seq)
            for i in range(len(self.annotation_names)):
                self.labels[self.annotation_names[i]].append(sequence["annotations"][i])

        self.seq_keypoints = np.array(seq_keypoints, dtype=np.float32)
        #self.items = list(np.arange(len(keypoints_ids)))
        #self.keypoints_ids = keypoints_ids
        #self.n_frames = len(self.keypoints_ids)
        for label_name in self.annotation_names:
            self.labels[label_name] = np.array(self.labels[label_name], dtype=np.float32)

        del self.raw_data

    # ------------------------------------------------------------
    def featurise_keypoints(self, keypoints):
        keypoints = self.normalize(keypoints)
        if self.centeralign:
            keypoints = keypoints.reshape(self.max_keypoints_len, *self.KEYFRAME_SHAPE)
            keypoints = self.transform_to_centeralign_components(keypoints)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        return keypoints