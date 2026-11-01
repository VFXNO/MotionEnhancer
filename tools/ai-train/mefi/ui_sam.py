"""Pinned EfficientSAM-Ti for offline label proposals, never runtime admission.

Downloading is an explicit operation. Inference reads only the selected frame;
no executable upstream code or additional packages are fetched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import urllib.request

import cv2
import numpy as np

from .ui_labeling import digest_bytes, file_hash


REVISION = '1e791988b34b2098736003057da0f3e6bc826259'
REPOSITORY = 'opencv/image_segmentation_efficientsam'
FILENAME = 'image_segmentation_efficientsam_ti_2025april.onnx'
MODEL_SHA256 = '4eb496e0a7259d435b49b66faf1754aa45a5c382a34558ddda9a8c6fe5915d77'
MODEL_BYTES = 48312857
LICENSE_SHA256 = 'c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4'
BASE_URL = f'https://huggingface.co/{REPOSITORY}/resolve/{REVISION}/'
MAX_CANDIDATES = 48


def model_folder():
    return Path(os.environ.get('LOCALAPPDATA', Path.home() / '.cache')) / 'MotionEnhancer/label-models/efficientsam-ti'


def model_path():
    return model_folder() / FILENAME


def download_file(url, path, expected, size, progress=None):
    """Verify before publishing; never overwrite an existing artifact."""
    if path.exists():
        if path.stat().st_size != size or file_hash(path) != expected:
            raise ValueError(f'Existing label-model file differs from the pin; it was not overwritten. '
                             f'Move it aside to a backup location, then use Verify / download again: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.download-', suffix='.partial', dir=path.parent)
    try:
        digest, received = hashlib.sha256(), 0
        with os.fdopen(fd, 'wb') as handle, urllib.request.urlopen(url, timeout=30) as response:
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                received += len(block)
                if received > size:
                    raise ValueError('Download exceeds its pinned size')
                digest.update(block)
                handle.write(block)
                if progress:
                    progress(received, size)
            handle.flush()
            os.fsync(handle.fileno())
        if received != size or digest.hexdigest() != expected:
            raise ValueError('Label-model download failed size/SHA256 verification')
        # Same-volume exclusive publication. Unlike replace(), this cannot
        # discard a file that appeared after the initial existence check.
        os.link(name, path)
    finally:
        if os.path.exists(name):
            os.remove(name)


def download_model(folder=None, progress=None):
    folder = Path(folder) if folder is not None else model_folder()
    download_file(BASE_URL + 'LICENSE', folder / 'LICENSE', LICENSE_SHA256, 11357)
    download_file(BASE_URL + FILENAME, folder / FILENAME, MODEL_SHA256, MODEL_BYTES, progress)
    metadata = {'repository': REPOSITORY, 'revision': REVISION, 'filename': FILENAME,
                'sha256': MODEL_SHA256, 'bytes': MODEL_BYTES, 'license': 'Apache-2.0',
                'license_sha256': LICENSE_SHA256, 'url': BASE_URL + FILENAME,
                'purpose': 'offline human-reviewed label suggestions only; not a game runtime model'}
    path = folder / 'source.json'
    if not path.exists():
        with path.open('x', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
    return folder / FILENAME


def prompt_arrays(shape, points=None, labels=None, box=None):
    """EfficientSAM labels: 0=negative, 1=positive, 2/3=box corners, -1=pad."""
    h, w = shape
    points, labels = list(points or []), list(labels or [])
    if any(type(label) is not int or label not in (0, 1) for label in labels):
        raise ValueError('Click labels must be 1 (positive) or 0 (negative); box corners are generated separately')
    if box is not None:
        if len(box) != 4 or not np.isfinite(box).all():
            raise ValueError('Invalid selection box')
        x0, y0, x1, y1 = box
        if not 0 <= x0 < x1 < w or not 0 <= y0 < y1 < h:
            raise ValueError('Draw a nonempty selection box inside the current frame')
        points = [[x0, y0], [x1, y1], *points]
        labels = [2, 3, *labels]
    if (not points or len(points) != len(labels) or len(points) > 6
            or any(type(label) is not int or label not in (0, 1, 2, 3) for label in labels)
            or not any(label in (1, 2) for label in labels)):
        raise ValueError('Select at least one positive point or a box; at most six total prompt points/corners')
    value = np.asarray(points, np.float32)
    if (value.shape != (len(points), 2) or not np.isfinite(value).all()
            or (value < 0).any() or (value[:, 0] >= w).any() or (value[:, 1] >= h).any()):
        raise ValueError('Prompt points must lie inside the prepared frame')
    coords = np.zeros((1, 1, 6, 2), np.float32)
    result_labels = np.full((1, 1, 6), -1, np.float32)
    coords[0, 0, :len(points)] = value * np.array([1024 / w, 1024 / h], np.float32)
    result_labels[0, 0, :len(points)] = labels
    return coords, result_labels


def mask_iou(a, b):
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.


def retain_candidates(candidates, limit=MAX_CANDIDATES, dedup=.85):
    """Stable score order, pixel-mask duplicate suppression; no box-filled labels."""
    kept = []
    for candidate in sorted(candidates, key=lambda item: item['score'], reverse=True):
        if any(mask_iou(candidate['mask'], old['mask']) >= dedup for old in kept):
            continue
        kept.append(candidate)
        if len(kept) >= limit:
            break
    return kept


class EfficientSam:
    def __init__(self, path=None):
        self.path = Path(path) if path is not None else model_path()
        if not self.path.is_file():
            raise ValueError('EfficientSAM-Ti is not downloaded. Use the Download model button first.')
        if self.path.stat().st_size != MODEL_BYTES:
            raise ValueError('EfficientSAM-Ti file size differs from the pin. Move the cached file to a backup location '
                             'and use Verify / download again; no file was replaced or inference run.')
        raw = self.path.read_bytes()
        if digest_bytes(raw) != MODEL_SHA256:
            raise ValueError('EfficientSAM-Ti checksum mismatch. Move the cached file to a backup location '
                             'and use Verify / download again; no file was replaced or inference run.')
        cv2.ocl.setUseOpenCL(False)
        cv2.setNumThreads(min(4, max(1, os.cpu_count() or 1)))
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(raw, np.uint8))
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.blob, self.shape = None, None

    def set_frame(self, frame):
        if (frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3
                or any(d < 1 or d > 4096 for d in frame.shape[:2])):
            raise ValueError('EfficientSAM expects an RGB uint8 frame up to 4096 pixels per dimension')
        self.shape = frame.shape[:2]
        image = cv2.resize(frame, (1024, 1024), interpolation=cv2.INTER_LINEAR)
        self.blob = np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255

    def predict(self, points=None, labels=None, box=None):
        if self.blob is None:
            raise ValueError('Select a frame before inference')
        coords, prompt_labels = prompt_arrays(self.shape, points, labels, box)
        self.net.setInput(self.blob, 'batched_images')
        self.net.setInput(coords, 'batched_point_coords')
        self.net.setInput(prompt_labels, 'batched_point_labels')
        logits, scores = self.net.forward(['output_masks', 'iou_predictions'])
        if (logits.shape != (1, 1, 3, 1024, 1024) or scores.shape != (1, 1, 3)
                or not np.isfinite(logits).all() or not np.isfinite(scores).all()):
            raise ValueError('EfficientSAM returned an invalid mask/quality tensor')
        h, w = self.shape
        results = []
        for index in np.argsort(-scores[0, 0], kind='stable'):
            # Match the upstream zero-logit threshold then nearest restore.
            # This is a shape candidate, not a calibrated UI probability map.
            mask = cv2.resize((logits[0, 0, index] >= 0).astype(np.uint8), (w, h),
                              interpolation=cv2.INTER_NEAREST).astype(bool)
            area = int(mask.sum())
            if not area:
                continue
            # Enforce explicit clicks in prepared-frame coordinates. Negative
            # points use label 0, not padding -1; no disk/background mutation.
            if any(bool(mask[int(y), int(x)]) != bool(label) for (x, y), label in zip(points or [], labels or [])):
                continue
            results.append({'mask': mask, 'score': float(scores[0, 0, index]), 'area': area,
                            'prompt': {'points': points or [], 'labels': labels or [], 'box': box}})
        return results

    def propose(self, *, points=None, labels=None, box=None, grid=None, progress=None):
        if self.shape is None:
            raise ValueError('Select a frame before generating candidates')
        limit = min(MAX_CANDIDATES, max(1, (64 * 1024 * 1024) // (self.shape[0] * self.shape[1])))
        if grid is None:
            candidates = self.predict(points, labels, box)
        else:
            if type(grid) is not int or grid not in (3, 5):
                raise ValueError('Automatic scan supports a bounded 3 x 3 or 5 x 5 point grid')
            candidates = []
            h, w = self.shape
            for index in range(grid * grid):
                point = [[min(w - 1, int((index % grid + .5) * w / grid)),
                          min(h - 1, int((index // grid + .5) * h / grid))]]
                for item in self.predict(point, [1]):
                    fraction = item['area'] / (h * w)
                    if .00005 <= fraction <= .80 and item['score'] >= .50:
                        candidates.append(item)
                # Bound memory even for large prepared frames and many prompts.
                candidates = retain_candidates(candidates, limit=limit)
                if progress:
                    progress(index + 1, grid * grid, len(candidates))
        candidates = retain_candidates(candidates, limit=limit)
        info = {'kind': 'efficientsam-candidates', 'model': str(self.path), 'sha256': MODEL_SHA256,
                'repository': REPOSITORY, 'revision': REVISION, 'license': 'Apache-2.0',
                'backend': 'OpenCV DNN CPU', 'resize': 'RGB 0..1 stretch to 1024; nearest binary mask restore',
                'grid': grid, 'min_predicted_iou': .50 if grid else None,
                'dedup_mask_iou': .85, 'max_candidates': limit,
                'role': 'unreviewed object masks; not automatic UI labels',
                'candidates': [{key: value for key, value in item.items() if key != 'mask'} for item in candidates]}
        masks = np.stack([item['mask'] for item in candidates]) if candidates else np.zeros((0, *self.shape), bool)
        return masks, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Explicitly download the pinned 48.3 MB labeling model')
    args = parser.parse_args()
    if args.download:
        path = download_model(progress=lambda done, total: print(f'{done}/{total}', flush=True))
        print(f'Ready: {path}', flush=True)
    else:
        print(model_path())


if __name__ == '__main__':
    main()
