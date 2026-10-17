"""Triplet datasets for training (built by mefi.dataset)."""
import json
import math
import os
import random
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def load_index(folders):
    """All triplets of one or more dataset folders: (path, video, entry)."""
    items = []
    for folder in folders:
        with open(os.path.join(folder, 'index.json'), encoding='utf-8') as f:
            for e in json.load(f):
                items.append((os.path.join(folder, 'triplets', e['dir']), f"{folder}|{e['video']}", e))
    return items


def split(items, val_fraction=0.1):
    """Validation by whole videos (a video's triplets look alike, so mixing
    them would measure memory, not generalisation). With fewer than three
    videos, the last part of each video is held out instead."""
    videos = sorted({v for _, v, _ in items})
    if len(videos) >= 3:
        val_videos = {v for v in videos if (zlib.crc32(v.encode()) % 1000) / 1000.0 < val_fraction}
        if not val_videos:
            val_videos = {videos[-1]}
        train = [i for i in items if i[1] not in val_videos]
        val = [i for i in items if i[1] in val_videos]
        return train, val
    train, val = [], []
    for v in videos:
        own = sorted((i for i in items if i[1] == v), key=lambda i: i[2]['frame'])
        cut = max(1, int(len(own) * (1.0 - val_fraction)))
        train += own[:cut]
        val += own[cut:]
    return train, val


def read(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'cannot read {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


class Triplets(Dataset):
    """Random crops with flips, time reversal and a little colour change
    (training), or fixed centre crops (validation).

    Training also adds motion the videos may not have:
      * a camera pan: the three crops are taken from windows shifted by -d,
        0 and +d, so the clip moves by exactly 2d from frame0 to frame2 on
        top of its own motion - linear, so the middle frame stays the true
        middle, and a pure offset, so nothing is resampled. Without large
        motion in the data the model learned to predict little motion and
        fell back to a blend wherever the game moved fast.
      * a zoom (0.6x to 1.6x) of all three frames, which scales every motion
        with it: the model sees each texture and speed at other scales, as
        the program shows it at a resolution the dataset was not built at.
    """

    PAN_PROBABILITY = 0.6
    ZOOM_PROBABILITY = 0.4

    def __init__(self, items, crop=256, train=True, pan=0.25):
        self.items = items
        # One size for every sample, a multiple of 16 (the network's three
        # scales): a batch cannot mix sizes. Crops used to shrink to fit each
        # frame, and a portrait video next to landscape ones then gave crops
        # of 192 and 352 px in one batch ("Trying to resize storage that is
        # not resizable").
        self.crop = max(64, crop // 16 * 16)
        self.train = train
        # Largest pan per frame step, a fraction of the crop (64 px of motion
        # from frame0 to frame2 at a 256 px crop by default).
        self.max_pan = int(self.crop * pan / 2)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        path, _, _ = self.items[k]
        ims = [read(os.path.join(path, f'im{i}.jpg')) for i in range(3)]
        c = self.crop
        if self.train and random.random() < self.ZOOM_PROBABILITY:
            z = math.exp(random.uniform(math.log(0.6), math.log(1.6)))
            h, w = ims[0].shape[:2]
            size = (max(8, int(round(w * z))), max(8, int(round(h * z))))
            interp = cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR
            ims = [cv2.resize(im, size, interpolation=interp) for im in ims]
        dx = dy = 0
        if self.train and self.max_pan > 0 and random.random() < self.PAN_PROBABILITY:
            dx = random.randint(-self.max_pan, self.max_pan)
            dy = random.randint(-self.max_pan, self.max_pan) // 2   # cameras pan more than tilt
        need_h, need_w = c + 2 * abs(dy), c + 2 * abs(dx)
        h, w = ims[0].shape[:2]
        if h < need_h or w < need_w:
            # Smaller than the crop (a portrait video, a low frame height):
            # all three scaled up together, so the motion stays consistent.
            scale = max(need_h / h, need_w / w)
            size = (max(need_w, int(math.ceil(w * scale))), max(need_h, int(math.ceil(h * scale))))
            ims = [cv2.resize(im, size, interpolation=cv2.INTER_LINEAR) for im in ims]
            h, w = ims[0].shape[:2]
        if self.train:
            # The middle frame's window, with room for the pan both ways.
            y = random.randint(abs(dy), h - c - abs(dy))
            x = random.randint(abs(dx), w - c - abs(dx))
        else:
            y, x = (h - c) // 2, (w - c) // 2
        offsets = ((-dx, -dy), (0, 0), (dx, dy))
        ims = [im[y + oy:y + oy + c, x + ox:x + ox + c] for im, (ox, oy) in zip(ims, offsets)]
        if self.train:
            if random.random() < 0.5:
                ims = [im[:, ::-1] for im in ims]
            if random.random() < 0.2:
                ims = [im[::-1] for im in ims]
            if random.random() < 0.5:
                ims = ims[::-1]          # time reversed: still a valid triplet
            if random.random() < 0.3:    # a little colour change, the same for all three
                gain = np.random.uniform(0.85, 1.15, 3).astype(np.float32)
                ims = [np.clip(im.astype(np.float32) * gain, 0, 255).astype(np.uint8) for im in ims]
        return torch.cat([to_tensor(im) for im in ims], 0)   # [9, c, c]


class FullFrames(Dataset):
    """Whole validation frames, cropped to multiples of 16 (for PSNR)."""

    def __init__(self, items, limit=64):
        step = max(1, len(items) // limit)
        self.items = items[::step][:limit]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        path, _, _ = self.items[k]
        ims = [read(os.path.join(path, f'im{i}.jpg')) for i in range(3)]
        h, w = ims[0].shape[:2]
        hh, ww = h // 16 * 16, w // 16 * 16
        return torch.cat([to_tensor(im[:hh, :ww]) for im in ims], 0)
