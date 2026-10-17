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


# A triplet whose 90th-percentile motion (dataset pixels) is below this is
# "still": a plain blend already rebuilds it, so it says nothing about how
# well the model follows motion and is left out of the headline score.
MOVING = 3.0


def split(items, val_fraction=0.1, gap=2):
    """The last part of every video is held out (in frame order, with `gap`
    triplets dropped before it, so the two sides are not neighbouring
    frames). Every video is represented on both sides.

    Holding out whole videos instead was measuring one arbitrary clip: with
    no video hashed into the validation share, the fallback took the last
    name alphabetically - on the user's data a nearly still live wallpaper,
    so the score sat at a plain blend's and the preview showed the same
    frame every epoch."""
    by_video = {}
    for item in items:
        by_video.setdefault(item[1], []).append(item)
    train, val = [], []
    for own in by_video.values():
        own.sort(key=lambda i: i[2]['frame'])
        if len(own) < 10:
            train += own
            continue
        held = max(1, int(round(len(own) * val_fraction)))
        train += own[:len(own) - held - gap]
        val += own[len(own) - held:]
    return train, val


def is_moving(item):
    return item[2].get('motion90', 0.0) >= MOVING


def read(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f'cannot read {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0


def crop_size(height):
    """(height, width) of a training crop: `height` and 16:9, the shape of
    the screen the model runs on, both multiples of 16 (352 -> 640 x 352,
    720 -> 1280 x 720)."""
    h = max(64, height // 16 * 16)
    return h, -(-h * 16 // (9 * 16)) * 16


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

    # Gentle on purpose: with a pan on 60% of samples, of up to a quarter of
    # the crop, so much of each crop slid out of view that the model fell
    # back to a plain blend (29.98 dB against 32.64 without pans, synthetic
    # test, 15 epochs).
    PAN_PROBABILITY = 0.3
    ZOOM_PROBABILITY = 0.4

    def __init__(self, items, crop=256, train=True, pan=0.15):
        self.items = items
        # One size for every sample, a multiple of 16 (the network's three
        # scales): a batch cannot mix sizes. Crops used to shrink to fit each
        # frame, and a portrait video next to landscape ones then gave crops
        # of 192 and 352 px in one batch ("Trying to resize storage that is
        # not resizable").
        self.crop_h, self.crop_w = crop_size(crop)
        self.train = train
        # Largest pan per frame step, a fraction of the crop height (38 px
        # of motion from frame0 to frame2 at a 256 px crop by default).
        self.max_pan = int(self.crop_h * pan / 2)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        path, _, _ = self.items[k]
        ims = [read(os.path.join(path, f'im{i}.jpg')) for i in range(3)]
        ch, cw = self.crop_h, self.crop_w
        if self.train and random.random() < self.ZOOM_PROBABILITY:
            z = math.exp(random.uniform(math.log(0.6), math.log(1.6)))
            h, w = ims[0].shape[:2]
            size = (max(8, int(round(w * z))), max(8, int(round(h * z))))
            interp = cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR
            ims = [cv2.resize(im, size, interpolation=interp) for im in ims]
        dx = dy = 0
        if self.train and self.max_pan > 0 and random.random() < self.PAN_PROBABILITY:
            # Mostly small, sometimes up to the largest (u squared), and
            # cameras pan more than they tilt.
            dx = int(round(self.max_pan * random.random() ** 2)) * random.choice((-1, 1))
            dy = int(round(self.max_pan * 0.5 * random.random() ** 2)) * random.choice((-1, 1))
        need_h, need_w = ch + 2 * abs(dy), cw + 2 * abs(dx)
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
            y = random.randint(abs(dy), h - ch - abs(dy))
            x = random.randint(abs(dx), w - cw - abs(dx))
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        offsets = ((-dx, -dy), (0, 0), (dx, dy))
        ims = [im[y + oy:y + oy + ch, x + ox:x + ox + cw] for im, (ox, oy) in zip(ims, offsets)]
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
        return torch.cat([to_tensor(im) for im in ims], 0)   # [9, ch, cw]


class FullFrames(Dataset):
    """Whole validation frames, cropped to multiples of 16 (for PSNR): up
    to `limit`, the moving ones first (is_moving), spread over the videos.
    Each sample also returns whether it is moving."""

    def __init__(self, items, limit=96):
        moving = [i for i in items if is_moving(i)]
        still = [i for i in items if not is_moving(i)]
        def spread(group, n):
            step = max(1, len(group) // max(1, n))
            return group[::step][:n]
        chosen = spread(moving, limit * 3 // 4)
        self.items = chosen + spread(still, limit - len(chosen))
        # The preview cycles through the most moving ones, a new one each epoch.
        ranked = sorted(range(len(self.items)), key=lambda k: -self.items[k][2].get('motion90', 0.0))
        self.preview_order = ranked[:24] or [0]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        path, _, _ = self.items[k]
        ims = [read(os.path.join(path, f'im{i}.jpg')) for i in range(3)]
        h, w = ims[0].shape[:2]
        hh, ww = h // 16 * 16, w // 16 * 16
        return torch.cat([to_tensor(im[:hh, :ww]) for im in ims], 0), torch.tensor(is_moving(self.items[k]))
