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
    (training), or fixed centre crops (validation)."""

    def __init__(self, items, crop=256, train=True):
        self.items = items
        # One size for every sample, a multiple of 16 (the network's three
        # scales): a batch cannot mix sizes. Crops used to shrink to fit each
        # frame, and a portrait video next to landscape ones then gave crops
        # of 192 and 352 px in one batch ("Trying to resize storage that is
        # not resizable").
        self.crop = max(64, crop // 16 * 16)
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        path, _, _ = self.items[k]
        ims = [read(os.path.join(path, f'im{i}.jpg')) for i in range(3)]
        h, w = ims[0].shape[:2]
        c = self.crop
        if h < c or w < c:
            # Smaller than the crop (a portrait video, a low frame height):
            # all three scaled up together, so the motion stays consistent.
            scale = c / min(h, w)
            size = (max(c, int(math.ceil(w * scale))), max(c, int(math.ceil(h * scale))))
            ims = [cv2.resize(im, size, interpolation=cv2.INTER_LINEAR) for im in ims]
            h, w = ims[0].shape[:2]
        if self.train:
            y = random.randint(0, h - c)
            x = random.randint(0, w - c)
        else:
            y, x = (h - c) // 2, (w - c) // 2
        ims = [im[y:y + c, x:x + c] for im in ims]
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
