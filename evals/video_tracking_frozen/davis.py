# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Reading DAVIS 2017 (Pont-Tuset et al., 2017) from its standard layout:

    <root>/JPEGImages/480p/<sequence>/00000.jpg ...
    <root>/Annotations/480p/<sequence>/00000.png ...   palette PNG, value = object id
    <root>/ImageSets/2017/val.txt                      one sequence name per line
"""

import os

import numpy as np
import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_sequences(root, split="val", year="2017"):
    path = os.path.join(root, "ImageSets", year, f"{split}.txt")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def sequence_files(root, sequence, resolution="480p"):
    """Sorted (frame paths, annotation paths) of one sequence."""
    frame_dir = os.path.join(root, "JPEGImages", resolution, sequence)
    mask_dir = os.path.join(root, "Annotations", resolution, sequence)
    frames = sorted(f for f in os.listdir(frame_dir) if f.endswith(".jpg"))
    masks = sorted(f for f in os.listdir(mask_dir) if f.endswith(".png"))
    if not frames or len(frames) != len(masks):
        raise ValueError(
            f"{sequence}: {len(frames)} frames but {len(masks)} annotations under {root}"
        )
    return (
        [os.path.join(frame_dir, f) for f in frames],
        [os.path.join(mask_dir, f) for f in masks],
    )


def read_annotation(path):
    """[H, W] uint8 object ids (0 = background)."""
    image = Image.open(path)
    if image.mode not in ("P", "L"):
        raise ValueError(
            f"{path} has mode {image.mode}; DAVIS annotations are palette PNGs "
            "whose pixel VALUE is the object id."
        )
    return np.array(image, dtype=np.uint8)


def model_input_size(height, width, short_side, patch_size):
    """Size the frames are resized to: short side `short_side`, aspect ratio
    kept, then each side rounded DOWN to a multiple of the patch size."""
    scale = short_side / min(height, width)
    new_h = max(patch_size, int(round(height * scale)) // patch_size * patch_size)
    new_w = max(patch_size, int(round(width * scale)) // patch_size * patch_size)
    return new_h, new_w


def load_frames(paths, short_side=480, patch_size=16):
    """[N, 3, H', W'] float32, ImageNet-normalised, plus the original (H, W)."""
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    frames, original = [], None
    for path in paths:
        image = Image.open(path).convert("RGB")
        if original is None:
            original = (image.height, image.width)
            size = model_input_size(image.height, image.width, short_side, patch_size)
        image = image.resize((size[1], size[0]), Image.BICUBIC)
        pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        frames.append((pixels.permute(2, 0, 1) - mean) / std)
    return torch.stack(frames), original
