import os
import numpy as np


def diff(vec, axis=-1, h=1, padding="edge"):
    assert padding in [
        "zero",
        "edge",
    ], "Padding must be one of ['zero', 'edge'],"
    " got {}.".format(padding)

    # move the target axis to the end
    vec = np.moveaxis(vec, axis, -1)

    # compute diff
    dvec = np.zeros_like(vec)
    dvec[..., h:] = vec[..., h:] - vec[..., :-h]

    # take care of padding the beginning
    if padding == "edge":
        for i in range(h):
            dvec[..., i] = dvec[..., h + 1]

    # move the axis back to its original position
    dvec = np.moveaxis(dvec, -1, axis)
    return dvec


def to_polar_coordinates(vec):
    r = np.linalg.norm(vec, axis=-1)
    theta = np.arctan2(vec[..., 1], vec[..., 0])
    return r, theta


def to_cartasian_coordinates(r, theta):
    x, y = r * np.cos(theta), r * np.sin(theta)
    return x, y


def angle_clip(theta):
    return np.mod(theta + np.pi, 2 * np.pi) - np.pi


"""Load mouse triplet data"""
def load_mice_triplet(path):
    # load raw train data (with annotations for 2 tasks)
    data_train = np.load(os.path.join(path, "mouse_triplet_train.npy"), allow_pickle=True).item()
    sequence_ids_train, sequence_data_train = zip(*data_train["sequences"].items())
    keypoints_train = np.stack([data["keypoints"] for data in sequence_data_train])

    # load submission data (no annoations)
    data_submission = np.load(os.path.join(path, "mouse_triplet_test.npy"), allow_pickle=True).item()
    sequence_ids_submission, sequence_data_submission = zip(*data_submission["sequences"].items())
    keypoints_submission = np.stack([data["keypoints"] for data in sequence_data_submission])

    # concatenate train and submission data
    sequence_ids = np.concatenate([sequence_ids_train, sequence_ids_submission], axis=0)
    keypoints = np.concatenate([keypoints_train, keypoints_submission], axis=0)

    split_mask = np.ones(len(sequence_ids), dtype=bool)
    split_mask[-len(sequence_ids_submission) :] = False

    # treat each mouse independently, keep track of which video each mouse came from
    num_samples, sequence_length, num_mice, num_keypoints, _ = keypoints.shape
    keypoints = keypoints.transpose((0, 2, 1, 3, 4))                       # (5336, 3, 1800, 12, 2)
    keypoints = keypoints.reshape((-1, sequence_length, num_keypoints, 2)) # (16008, 1800, 12, 2)
    batch = np.repeat(np.arange(num_samples), num_mice)

    return keypoints, split_mask, batch


"""Extract mouse features from keypoints"""
def mouse_feature_extractor(keypoints, noise_thresh=3e-3):
    # compute state features
    # body part 1: head, keypoints 0, 1, 2, 3
    head_center = keypoints[..., 3, :]
    head_orientation = np.arctan2(
        keypoints[..., 0, 1] - keypoints[..., 3, 1],
        keypoints[..., 0, 0] - keypoints[..., 3, 0],
    )
    # body part 2: forepaws, keypoints 3, 4, 5
    # use keypoint 3 as center
    left_forepaw = keypoints[..., 4, :] - keypoints[..., 3, :]
    right_forepaw = keypoints[..., 5, :] - keypoints[..., 3, :]

    left_forepaw_r, left_forepaw_theta = to_polar_coordinates(left_forepaw)
    right_forepaw_r, right_forepaw_theta = to_polar_coordinates(right_forepaw)
    forepaws_theta = angle_clip(right_forepaw_theta - left_forepaw_theta)

    # connection body parts 2-3
    spine = keypoints[..., 6, :] - keypoints[..., 3, :]
    spine_r, spine_theta = to_polar_coordinates(spine)

    # body part 3: bottom, keypoints 6, 7, 8, 9
    bottom_center = keypoints[..., 6, :]
    # center
    bottom = keypoints[..., 7:, :] - bottom_center[..., np.newaxis, :]
    bottom_orientation = np.arctan2(
        keypoints[..., 6, 1] - keypoints[..., 9, 1],
        keypoints[..., 6, 0] - keypoints[..., 9, 0],
    )
    bottom_rotation = np.array(
        [
            [np.cos(-bottom_orientation), -np.sin(-bottom_orientation)],
            [np.sin(-bottom_orientation), np.cos(-bottom_orientation)],
        ]
    )
    # rotate
    bottom = np.einsum("ijkp,lpij->ijkl", bottom, bottom_rotation)

    left_hindpaw_r, left_hindpaw_theta = to_polar_coordinates(bottom[..., 0, :])
    left_hindpaw_theta = left_hindpaw_theta
    right_hindpaw_r, right_hindpaw_theta = to_polar_coordinates(bottom[..., 1, :])
    right_hindpaw_theta = right_hindpaw_theta
    center_to_tail_r, _ = to_polar_coordinates(bottom[..., 2, :])

    _, tail_theta_1 = to_polar_coordinates(bottom[..., 3, :] - bottom[..., 2, :])
    tail_theta_1 = tail_theta_1
    _, tail_theta_2 = to_polar_coordinates(bottom[..., 4, :] - bottom[..., 3, :])
    tail_theta_2 = tail_theta_2

    # compute action features
    ### body part 1: head
    head_vx = diff(head_center[..., 0])
    head_vy = diff(head_center[..., 0])
    head_vr, head_vtheta = to_polar_coordinates(np.stack([head_vx, head_vy], axis=-1))
    head_vtheta[head_vr < noise_thresh] = 0.0
    head_vr[head_vr < noise_thresh] = 0.0
    head_dvtheta = angle_clip(diff(head_vtheta))
    # orientation
    head_orientation_dtheta = angle_clip(diff(head_orientation))
    ### body part 2: forepaws
    # left forepaw
    left_forepaw_dr = diff(left_forepaw_r)
    left_forepaw_dtheta = angle_clip(diff(left_forepaw_theta))
    # right forepaw
    right_forepaw_dr = diff(left_forepaw_r)
    right_forepaw_dtheta = angle_clip(diff(right_forepaw_theta))
    # angle between forepaws
    forepaws_dtheta = angle_clip(diff(forepaws_theta))
    # body part 3: bottom
    # velocity
    bottom_vx = diff(bottom_center[..., 0])
    bottom_vy = diff(bottom_center[..., 1])
    bottom_vr, bottom_vtheta = to_polar_coordinates(np.stack([bottom_vx, bottom_vy], axis=-1))
    bottom_vtheta[bottom_vr < noise_thresh] = 0.0
    bottom_vr[bottom_vr < noise_thresh] = 0.0
    bottom_dvtheta = angle_clip(diff(bottom_vtheta))
    # orientation
    bottom_orientation_dtheta = angle_clip(diff(bottom_orientation))
    # left hindpaw
    left_hindpaw_dr = diff(left_hindpaw_r)
    left_hindpaw_dtheta = angle_clip(diff(left_hindpaw_theta))
    # right hindpaw
    right_hindpaw_dr = diff(right_hindpaw_r)
    right_hindpaw_dtheta = angle_clip(diff(right_hindpaw_theta))
    # body part 4: tail
    tail_dtheta_1 = angle_clip(diff(tail_theta_1))
    tail_dtheta_2 = angle_clip(diff(tail_theta_2))
    # connections between body parts
    center_to_tail_dr = diff(center_to_tail_r)
    spine_dr = diff(spine_r)
    spine_dtheta = angle_clip(diff(spine_theta))

    ignore_frames = np.any(keypoints[..., 0] == 0, axis=-1)
    ignore_frames[:, 1:] = np.logical_or(ignore_frames[:, 1:], ignore_frames[:, :-1])

    input_features = np.stack(
        [
            head_center[..., 0], head_center[..., 1],
            np.cos(head_orientation), np.sin(head_orientation),
            left_forepaw_r,
            np.cos(left_forepaw_theta), np.sin(left_forepaw_theta),
            right_forepaw_r,
            np.cos(right_forepaw_theta), np.sin(right_forepaw_theta),
            np.cos(forepaws_theta), np.sin(forepaws_theta),
            bottom_center[..., 0], bottom_center[..., 1],
            np.cos(bottom_orientation), np.sin(bottom_orientation),
            left_hindpaw_r,
            np.cos(left_hindpaw_theta), np.sin(left_hindpaw_theta),
            right_hindpaw_r,
            np.cos(right_hindpaw_theta), np.sin(right_hindpaw_theta),
            center_to_tail_r,
            np.cos(tail_theta_1), np.sin(tail_theta_1),
            np.cos(tail_theta_2), np.sin(tail_theta_2),
            spine_r,
            np.cos(spine_theta), np.sin(spine_theta),
            head_vr,
            np.cos(head_vtheta), np.sin(head_vtheta),
            np.cos(head_dvtheta), np.sin(head_dvtheta),
            np.cos(head_orientation_dtheta), np.sin(head_orientation_dtheta),
            left_forepaw_dr,
            np.cos(left_forepaw_dtheta),np.sin(left_forepaw_dtheta),
            right_forepaw_dr,
            np.cos(right_forepaw_dtheta), np.sin(right_forepaw_dtheta),
            np.cos(forepaws_dtheta), np.sin(forepaws_dtheta),
            bottom_vr,
            np.cos(bottom_vtheta),  np.sin(bottom_vtheta),
            np.cos(bottom_dvtheta), np.sin(bottom_dvtheta),
            np.cos(bottom_orientation_dtheta), np.sin(bottom_orientation_dtheta),
            left_hindpaw_dr,
            np.cos(left_hindpaw_dtheta), np.sin(left_hindpaw_dtheta),
            right_hindpaw_dr,
            np.cos(right_hindpaw_dtheta), np.sin(right_hindpaw_dtheta),
            np.cos(tail_dtheta_1), np.sin(tail_dtheta_1),
            np.cos(tail_dtheta_2), np.sin(tail_dtheta_2),
            center_to_tail_dr,
            spine_dr,
            np.cos(spine_dtheta), np.sin(spine_dtheta),
            ignore_frames,
        ],
        axis=-1,
    )
    target_feats = np.stack(
        [
            head_vr,
            head_vtheta,
            head_dvtheta,
            head_orientation_dtheta,
            bottom_vr,
            bottom_vtheta,
            bottom_dvtheta,
            bottom_orientation_dtheta,
            spine_dr,
        ], axis=-1,
    )
    return input_features, target_feats, ignore_frames
