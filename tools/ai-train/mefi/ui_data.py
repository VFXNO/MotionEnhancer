"""Explicit dense UI labels only. No builder, OCR, teacher or pseudo-labels."""
import hashlib
import json
from pathlib import Path
import random

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .data import crop_size, to_tensor


LABEL_FORMAT = 'me-ui-labels-1'
LOCKED_CAPTURE = Path(__file__).resolve().parents[3] / 'build-project/Release/captures/20261101-124214'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_reference(root, value):
    if not isinstance(value, dict) or set(value) != {'path', 'sha256'}:
        raise ValueError('Each UI file needs exactly path and sha256')
    path, digest = value['path'], value['sha256']
    if not isinstance(path, str) or not path or Path(path).is_absolute():
        raise ValueError('UI paths must be relative to the manifest directory')
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError('UI file escapes the manifest directory')
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('UI files require lowercase SHA256 hashes')
    return {'path': resolved, 'sha256': digest}


class Labels:
    """Validate an allow-list and explicit source-disjoint splits.

    Test entries participate in leakage checks, but their files are NEVER read.
    Hash verification reads only explicitly listed training/validation files.
    Source IDs must identify the original recording, not a crop or renamed copy.
    """
    def __init__(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        manifest = json.loads(raw.decode('utf-8'))
        if not isinstance(manifest, dict) or manifest.get('format') != LABEL_FORMAT or not isinstance(manifest.get('label_version'), str) or not manifest['label_version']:
            raise ValueError('Expected me-ui-labels-1 and a nonempty label_version')
        if not isinstance(manifest.get('samples'), list):
            raise ValueError('UI manifest needs an explicit samples list')
        self.path, self.items = path, {'train': [], 'val': [], 'test': []}
        sources, frame_hashes, ids = {}, {}, set()
        for value in manifest['samples']:
            if not isinstance(value, dict):
                raise ValueError('Invalid UI sample')
            name, source, split = value.get('id'), value.get('source'), value.get('split')
            if not isinstance(name, str) or not name or name in ids:
                raise ValueError('UI sample IDs must be nonempty and unique')
            ids.add(name)
            if not isinstance(source, str) or not source or split not in self.items:
                raise ValueError('Each UI sample needs source and split=train/val/test')
            if source in sources and sources[source] != split:
                raise ValueError('UI source leakage: an original recording crosses splits')
            sources[source] = split
            h, w = value.get('height'), value.get('width')
            if any(type(d) is not int or d < 64 or d % 16 for d in (h, w)):
                raise ValueError('UI dimensions must be multiples of 16, at least 64')
            frame_ids = value.get('frame_ids')
            if (not isinstance(frame_ids, list) or len(frame_ids) != 3
                    or any(type(i) is not int or i < 0 for i in frame_ids)
                    or frame_ids != list(range(frame_ids[0], frame_ids[0] + 3))):
                raise ValueError('UI windows need three consecutive, chronological frame_ids')
            refs = value.get('frames')
            if not isinstance(refs, list) or len(refs) != 3:
                raise ValueError('UI samples need exactly three RGB frames')
            frames = [file_reference(path.parent, ref) for ref in refs]
            target = file_reference(path.parent, value.get('coverage'))
            valid = file_reference(path.parent, value.get('valid'))
            if target['path'].suffix.lower() != '.png' or valid['path'].suffix.lower() != '.png':
                raise ValueError('Dense coverage/valid labels must be lossless grayscale PNGs')
            if len({f['path'] for f in frames}) != 3:
                raise ValueError('A temporal UI window cannot repeat a frame path')
            for ref in frames:
                digest = ref['sha256']
                if digest in frame_hashes and frame_hashes[digest] != split:
                    raise ValueError('UI frame-copy leakage across splits')
                frame_hashes[digest] = split
            if split != 'test' and any(ref['path'].is_relative_to(LOCKED_CAPTURE.resolve()) for ref in [*frames, target, valid]):
                raise ValueError('The locked Arabic capture is test-only and may not be prepared or trained here')
            self.items[split].append({'id': name, 'source': source, 'frame_ids': frame_ids,
                                      'height': h, 'width': w, 'frames': frames,
                                      'coverage': target, 'valid': valid})
        if not self.items['train'] or not self.items['val']:
            raise ValueError('UI training requires explicit nonempty source-disjoint train and val splits')
        verified = set()
        for split in ('train', 'val'):
            for item in self.items[split]:
                for ref in [*item['frames'], item['coverage'], item['valid']]:
                    key = (ref['path'], ref['sha256'])
                    if key not in verified:
                        if sha256(ref['path']) != ref['sha256']:
                            raise ValueError(f'UI file SHA256 mismatch: {ref["path"]}')
                        verified.add(key)
        self.identity = {'format': LABEL_FORMAT, 'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                         'label_version': manifest['label_version'],
                         'counts': {key: len(value) for key, value in self.items.items()}}
        self.sources = set(sources)

    def require_flow_disjoint(self, flow_items):
        """Conservative: UI recordings cannot also occur in ANY flow split.

        The original flow splitter/targets remain unchanged. Source-aware flow
        indices are required; renamed/related recordings still need human review.
        """
        for _, source, _ in flow_items:
            if not source.startswith('source:'):
                raise ValueError('Joint UI training requires source-aware flow indices; flow-only training is unchanged')
            if source[len('source:'):] in self.sources:
                raise ValueError('UI and flow recordings must be disjoint, including their held-out/test sources')


def decode(ref):
    raw = ref['path'].read_bytes()
    if hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError(f'UI file changed after preflight: {ref["path"]}')
    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None or image.dtype != np.uint8:
        raise ValueError(f'UI files must decode to 8-bit images: {ref["path"]}')
    return image


def head_labels(target, valid):
    """Aligned 2x2 coverage; any cell touching unknown pixels is ignored."""
    target, valid = target[None], valid[None]
    known = F.avg_pool2d(valid, 2, 2) == 1
    coverage = F.avg_pool2d(target * valid, 2, 2)
    return torch.where(known, coverage, torch.zeros_like(coverage))[0], known.float()[0]


class UiFrames:
    def __init__(self, labels, split, crop=None):
        if split not in ('train', 'val'):
            raise ValueError('This trainer never loads the locked UI test split')
        self.items = labels.items[split]
        self.crop = crop_size(crop) if crop is not None else None
        if self.crop and any(i['height'] < self.crop[0] or i['width'] < self.crop[1] for i in self.items):
            raise ValueError('UI frames must fit the selected crop; no automatic text/image resizing')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        seed = None
        if isinstance(index, tuple):
            index, seed = index
        item = self.items[index]
        frames = [decode(ref) for ref in item['frames']]
        target, valid = decode(item['coverage']), decode(item['valid'])
        shape = (item['height'], item['width'])
        if any(frame.shape != (*shape, 3) for frame in frames) or target.shape != shape or valid.shape != shape:
            raise ValueError('UI frames and grayscale labels must match the declared full-frame dimensions')
        if not np.isin(valid, [0, 255]).all():
            raise ValueError('UI valid PNG must contain only 0=unknown or 255=known')
        if self.crop:
            h, w = self.crop
            rng = random.Random(seed if seed is not None else 0)
            y = 2 * rng.randint(0, (shape[0] - h) // 2)
            x = 2 * rng.randint(0, (shape[1] - w) // 2)
            frames = [im[y:y + h, x:x + w] for im in frames]
            target, valid = target[y:y + h, x:x + w], valid[y:y + h, x:x + w]
        target = torch.from_numpy(np.ascontiguousarray(target))[None].float() / 255
        valid = torch.from_numpy(np.ascontiguousarray(valid))[None].float() / 255
        target, valid = head_labels(target, valid)
        if not bool(valid.any()):
            raise ValueError('UI sample/crop contains no fully known head-resolution cells; adjust the labels/crop')
        frames = torch.cat([to_tensor(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)) for im in frames], 0)
        return frames, target, valid


def batch_at_step(dataset, step, batch, seed):
    """Independent deterministic UI stream, resumable without decoding old batches."""
    from .data import EpochBatches
    sampler = EpochBatches(len(dataset), batch, seed)
    sampler.epoch, sampler.start_batch = divmod(step, sampler.steps)
    keys = next(iter(sampler))
    samples = [dataset[key] for key in keys]
    return tuple(torch.stack([sample[i] for sample in samples]) for i in range(3))
