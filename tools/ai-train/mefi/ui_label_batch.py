"""Bounded batch preparation of explicitly selected video windows, not labels.

One sequential decode preserves exact frame ordinals; only selected triplets
are retrieved as color images. Results stay in the operation's temporary
directory until the editor accepts a completed import.
"""
from collections import deque
import math
from pathlib import Path

import cv2
from .ui_labeling import digest_bytes, file_hash, permitted_source, prepare_frames, source_id


MAX_WINDOWS = 100
MAX_BATCH_BYTES = 512 * 1024 * 1024


def window_positions(first, last, count):
    if type(first) is not int or type(last) is not int or first < 2 or last < first:
        raise ValueError('Choose a current-frame range starting at 2 or later')
    if type(count) is not int or not 1 <= count <= MAX_WINDOWS:
        raise ValueError(f'Choose 1..{MAX_WINDOWS} windows per operation')
    # Do not manufacture lots of overlapping examples from a tiny clip.
    count = min(count, 1 + (last - first) // 3)
    if count == 1:
        return [first]
    return [first + i * (last - first) // (count - 1) for i in range(count)]


def prepare_video_batch(path, root, *, count=20, first=2, last=None, resize=None, progress=None):
    path, root = permitted_source(path), Path(root).resolve()
    if type(count) is not int or not 1 <= count <= MAX_WINDOWS or type(first) is not int or first < 2:
        raise ValueError('Choose 1..100 windows and a first current frame of at least 2')
    before = path.stat()
    digest = file_hash(path)
    cap = cv2.VideoCapture(str(path))
    records, warnings, retained = [], [], 0
    try:
        if not cap.isOpened():
            raise ValueError('The installed video decoder could not open this file')
        estimated = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if last is None:
            if not math.isfinite(estimated) or estimated < first + 1 or estimated > 10000001:
                raise ValueError('This video has no usable frame-count metadata; enter an explicit last current frame')
            last = int(estimated) - 1
        if type(last) is not int or last > 10000000:
            raise ValueError('Last current frame must be at most 10,000,000')
        positions = window_positions(first, last, count)
        targets = set(positions)
        needed = {index for current in positions for index in range(current - 2, current + 1)}
        ring = deque(maxlen=3)
        end = positions[-1]
        for index in range(end + 1):
            # grab() still advances the inter-frame decoder exactly once. Do
            # not retrieve/convert thousands of unused frames to BGR images.
            if not cap.grab():
                warnings.append('Decoder ended before the requested final frame; only fully decoded windows were prepared.')
                break
            if index in needed:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f'Could not retrieve decoded video frame {index}; no windows accepted')
                ring.append(frame)
            if progress and index % 120 == 0:
                progress(index, end, len(records), len(positions))
            if index not in targets:
                continue
            frames = prepare_frames([cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in ring], resize)
            # Bound raw copies as well as encoded bytes, keeping large native
            # frames from producing a multi-gigabyte staging operation.
            bytes_needed = sum(im.nbytes for im in frames)
            if retained + bytes_needed > MAX_BATCH_BYTES:
                warnings.append('The 512 MiB batch limit was reached; use fewer windows or smaller prepared copies.')
                break
            encoded = []
            for frame in frames:
                ok, data = cv2.imencode('.png', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if not ok:
                    raise ValueError('Could not encode prepared video frame')
                encoded.append(data.tobytes())
            retained_bytes = max(bytes_needed, sum(map(len, encoded)))
            if retained + retained_bytes > MAX_BATCH_BYTES:
                warnings.append('The 512 MiB batch limit was reached; use fewer windows or smaller prepared copies.')
                break
            ordinal, refs = len(records), []
            for i, raw in enumerate(encoded):
                name = f'windows/{ordinal}/{i}.png'
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('xb') as handle:
                    handle.write(raw)
                refs.append({'path': name, 'sha256': digest_bytes(raw)})
            retained += retained_bytes
            h, w = frames[0].shape[:2]
            records.append({'height': h, 'width': w, 'frames': refs, 'frame_ids': [index - 2, index - 1, index],
                            'provenance': {'kind': 'video', 'path': str(path), 'sha256': digest,
                                           'source_id': source_id(path), 'resize': resize,
                                           'original_shape': list(ring[0].shape), 'frame_ids': [index - 2, index - 1, index],
                                           'sampling': 'evenly spaced requested ordinals, sequential grab; retrieve selected triplets only'}})
        if not records:
            raise ValueError('No complete window fit this video/range/batch limit; choose an earlier range or smaller prepared size')
    finally:
        cap.release()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source video changed during batch preparation; no windows accepted')
    if len(records) < count and not warnings:
        warnings.append('The requested range is too short for that many non-overlapping windows; fewer were prepared.')
    return {'kind': 'video-batch', 'requested': count, 'prepared': len(records), 'first': first, 'last': last,
            'warnings': warnings, 'samples': records}
