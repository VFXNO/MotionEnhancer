"""Triplet datasets for training (built by mefi.dataset)."""
import hashlib
import json
import math
import os
import random
import zlib

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def load_index(folders):
    """All triplets of one or more dataset folders: (path, video, entry),
    and how many the index lists whose images are gone (a deleted
    `triplets` folder): those are left out instead of failing mid-epoch."""
    items = []
    missing = 0
    seen_folders, seen_samples = set(), set()
    for folder in folders:
        folder = os.path.normcase(os.path.realpath(folder))
        if folder in seen_folders:
            continue
        seen_folders.add(folder)
        with open(os.path.join(folder, 'index.json'), encoding='utf-8') as f:
            for e in json.load(f):
                path = os.path.join(folder, 'triplets', e['dir'])
                if not all(os.path.exists(os.path.join(path, f'im{i}.jpg')) for i in range(3)):
                    missing += 1
                    continue
                # A source keeps its identity across dataset folders. Otherwise
                # copied/rebuilt triplets can be counted twice or leak across
                # independent temporal splits. Legacy indices stay readable.
                video = f"source:{e['source']}" if e.get('source') else f"{folder}|{e['video']}"
                sample = (video, e['frame'], e.get('stride', 1))
                if sample in seen_samples:
                    continue
                seen_samples.add(sample)
                items.append((path, video, e))
    return items, missing


# A triplet whose 90th-percentile motion (dataset pixels) is below this is
# "still" for the motion-only headline. This does not mean an easy target:
# a flash can have identical endpoints and a very different middle frame.
# Those samples still train normally and count in the all-sample score.
MOVING = 3.0
VALIDATION_PROTOCOL = 'temporal-balanced-v1'


def validation_identity(dataset):
    """Identify the scored sample selection, so resumed best scores compare
    like with like. Does not read or change the user's frame files."""
    return {'protocol': VALIDATION_PROTOCOL, 'selection': sample_identity(dataset.items)}


def sample_identity(items):
    samples = []
    for item in items:
        _, video, e = item
        samples.append((video, e['frame'], e.get('stride', 1), e.get('width'), e.get('height'), is_moving(item)))
    return hashlib.sha256(json.dumps(samples, ensure_ascii=True).encode('utf-8')).hexdigest()


class EpochBatches(Sampler):
    """Deterministic order and augmentation seeds, independent of worker
    prefetch. Resume starts at the first unconsumed batch, without decoding
    or augmenting the samples that already trained."""

    def __init__(self, size, batch, seed):
        self.size, self.batch, self.seed = size, batch, seed
        self.steps = max(1, size // batch)
        self.epoch, self.start_batch = 0, 0

    def __len__(self):
        return max(0, self.steps - self.start_batch)

    def __iter__(self):
        seed = (self.seed + self.epoch * 1_000_003) % (2 ** 63)
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(self.size, generator=generator).tolist()
        for batch in range(self.start_batch, self.steps):
            yield [(index, (seed + index * 97_409) % (2 ** 63))
                   for index in order[batch * self.batch:(batch + 1) * self.batch]]


def split(items, val_fraction=0.1, gap=2):
    """The last part of every video is held out (in frame order, with `gap`
    triplets dropped before it, plus removal of any triplet whose source
    frame interval reaches the held-out section). Longer-stride triplets
    can overlap even when several entries separate them in the index.
    Videos with fewer than ten samples are used only for training.

    Holding out whole videos instead was measuring one arbitrary clip: with
    no video hashed into the validation share, the fallback took the last
    name alphabetically - on the user's data a nearly still live wallpaper,
    so the score sat at a plain blend's and the preview showed the same
    frame every epoch."""
    if not 0 < val_fraction < 1 or gap < 0:
        raise ValueError('validation fraction must be between 0 and 1, and gap must be nonnegative')
    by_video = {}
    for item in items:
        by_video.setdefault(item[1], []).append(item)
    train, val = [], []
    for own in by_video.values():
        own.sort(key=lambda i: (i[2]['frame'], i[2].get('stride', 1), i[0]))
        if len(own) < 10:
            train += own
            continue
        held = min(len(own) - 1, max(1, int(round(len(own) * val_fraction))))
        boundary = own[len(own) - held][2]['frame']
        train += [i for i in own[:max(0, len(own) - held - gap)]
                  if i[2]['frame'] + 2 * i[2].get('stride', 1) < boundary]
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
        if isinstance(k, tuple):
            k, seed = k
            rng, colors = random.Random(seed), np.random.default_rng(seed)
        else:
            rng, colors = random, np.random
        path, _, _ = self.items[k]
        ims = [read(os.path.join(path, f'im{i}.jpg')) for i in range(3)]
        ch, cw = self.crop_h, self.crop_w
        if self.train and rng.random() < self.ZOOM_PROBABILITY:
            z = math.exp(rng.uniform(math.log(0.6), math.log(1.6)))
            h, w = ims[0].shape[:2]
            size = (max(8, int(round(w * z))), max(8, int(round(h * z))))
            interp = cv2.INTER_AREA if z < 1 else cv2.INTER_LINEAR
            ims = [cv2.resize(im, size, interpolation=interp) for im in ims]
        dx = dy = 0
        if self.train and self.max_pan > 0 and rng.random() < self.PAN_PROBABILITY:
            # Mostly small, sometimes up to the largest (u squared), and
            # cameras pan more than they tilt.
            dx = int(round(self.max_pan * rng.random() ** 2)) * rng.choice((-1, 1))
            dy = int(round(self.max_pan * 0.5 * rng.random() ** 2)) * rng.choice((-1, 1))
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
            y = rng.randint(abs(dy), h - ch - abs(dy))
            x = rng.randint(abs(dx), w - cw - abs(dx))
        else:
            y, x = (h - ch) // 2, (w - cw) // 2
        offsets = ((-dx, -dy), (0, 0), (dx, dy))
        ims = [im[y + oy:y + oy + ch, x + ox:x + ox + cw] for im, (ox, oy) in zip(ims, offsets)]
        if self.train:
            if rng.random() < 0.5:
                ims = [im[:, ::-1] for im in ims]
            if rng.random() < 0.2:
                ims = [im[::-1] for im in ims]
            if rng.random() < 0.5:
                ims = ims[::-1]          # time reversed: still a valid triplet
            if rng.random() < 0.3:    # a little colour change, the same for all three
                gain = colors.uniform(0.85, 1.15, 3).astype(np.float32)
                ims = [np.clip(im.astype(np.float32) * gain, 0, 255).astype(np.uint8) for im in ims]
        return torch.cat([to_tensor(im) for im in ims], 0)   # [9, ch, cw]


class FullFrames(Dataset):
    """Whole validation frames, cropped to multiples of 16 (for PSNR): up
    to `limit`, the moving ones first (is_moving), spread over the videos.
    Each sample also returns whether it is moving."""

    def __init__(self, items, limit=96):
        if limit < 1:
            raise ValueError('validation sample limit must be positive')
        moving = [i for i in items if is_moving(i)]
        still = [i for i in items if not is_moving(i)]
        def spread(group, n):
            # Allocate evenly between source videos, then span the full held-
            # out interval of each. A long recording must not crowd smaller
            # games/clips out of the score used to choose best.pt.
            videos = {}
            for entry in group:
                videos.setdefault(entry[1], []).append(entry)
            videos = {v: sorted(videos[v], key=lambda i: (i[2]['frame'], i[2].get('stride', 1), i[0]))
                      for v in sorted(videos)}
            quotas = dict.fromkeys(videos, 0)
            remaining = min(n, len(group))
            while remaining:
                for v, entries in videos.items():
                    if quotas[v] < len(entries) and remaining:
                        quotas[v] += 1
                        remaining -= 1
            chosen = []
            for v, entries in videos.items():
                count = quotas[v]
                for k in range(count):
                    index = len(entries) // 2 if count == 1 else k * (len(entries) - 1) // (count - 1)
                    chosen.append(entries[index])
            return chosen
        n_moving = min(len(moving), max(1, limit * 3 // 4))
        n_still = min(len(still), limit - n_moving)
        # If there are not enough stills, use the remaining moving clips
        # instead of silently discarding a quarter of the validation budget.
        n_moving = min(len(moving), limit - n_still)
        self.items = spread(moving, n_moving) + spread(still, n_still)
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
