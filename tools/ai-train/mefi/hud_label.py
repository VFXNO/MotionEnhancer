"""HUD labeling backend for the Label UI page (mefi.hud_label_gui).

Three parts:

* read_window: a short run of consecutive frames from a video, ending at
  the frame being labeled;
* suggest: where the HUD is, from how those frames behave - the same idea
  as the program's HUD persistence (shaders/UiPersistCS.hlsl): edges that
  stay at the same screen position, with the same gradient, while the scene
  around them moves. It pre-paints the label; the user corrects it;
* Project: a folder holding the reviewed samples and their manifest in the
  format the UI trainer reads (mefi.ui_data, me-ui-labels-1): for each
  sample three consecutive frames, a coverage PNG (255 = HUD) and a valid
  PNG (255 = known, 0 = unknown), every file listed with its SHA256.
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from .ui_data import LABEL_FORMAT, LOCKED_CAPTURE

PROJECT_FILE = 'labels.json'
SPLITS = ('train', 'val', 'test')

# Suggestion (see UiPersistCS.hlsl for the reasoning behind each value).
CELL = 4
EDGE_LEVEL = 6.0          # RMS luma gradient per pixel, 8-bit levels
STRUCTURE_CHANGE = 0.1    # gradient change / gradient energy of a kept edge pixel
STEADY_PIXELS = 2         # kept edge pixels a cell needs (of 16)
CONTRAST_LEVEL = 24.0     # luma above/below its 5x5 surroundings that makes a pixel stand out
CHANGE_LEVEL = 32.0       # a changed cell (above the GPU's 24: video is compressed)
MOTION_LEVEL = 3.0        # mean luma change of a cell that counts as scene motion
RING_CELLS = 6            # 24 px: where "the scene around" is looked at
RISE, FALL = 0.1, 0.25
STATIC_LEVEL = 20.0       # the largest change a suggested HUD pixel may show (compression)


# ---------------------------------------------------------------- frames

def video_info(path):
    """Frame count, frames per second, width, height."""
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        return (int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
                int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    finally:
        capture.release()


def read_window(path, last, count):
    """`count` consecutive BGR frames ending at frame `last` (or fewer at the
    start of the video), read in order after one seek so they really are
    consecutive."""
    first = max(0, last - count + 1)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f'Cannot open video: {path}')
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, first)
        frames = []
        for _ in range(last - first + 1):
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if len(frames) < 3:
        raise ValueError(f'Could not read three frames ending at frame {last}')
    return first, frames


def label_positions(frame_count, samples, margin=30):
    """Frames to label, spread evenly over the video (each needs `margin`
    frames before it for the suggestion)."""
    first, last = min(margin, frame_count - 1), frame_count - 2
    if last <= first:
        return [max(2, frame_count - 1)]
    samples = max(1, min(samples, last - first + 1))
    return sorted({int(round(first + (last - first) * (i + 0.5) / samples)) for i in range(samples)})


# ---------------------------------------------------------------- suggestion

def _pool(values, reduce):
    h, w = values.shape[:2]
    ch, cw = h // CELL, w // CELL
    v = values[:ch * CELL, :cw * CELL].reshape(ch, CELL, cw, CELL)
    return reduce(v, axis=(1, 3))


def _gradient(luma):
    gx = np.zeros_like(luma)
    gy = np.zeros_like(luma)
    gx[:, 1:-1] = 0.5 * (luma[:, 2:] - luma[:, :-2])
    gy[1:-1, :] = 0.5 * (luma[2:, :] - luma[:-2, :])
    return gx, gy


def _ring_max(values):
    """The largest value on a ring of eight cells RING_CELLS away."""
    h, w = values.shape
    padded = np.pad(values, RING_CELLS, mode='edge')
    out = np.zeros_like(values)
    for k in range(8):
        dy = int(round(RING_CELLS * np.sin(k * np.pi / 4)))
        dx = int(round(RING_CELLS * np.cos(k * np.pi / 4)))
        out = np.maximum(out, padded[RING_CELLS + dy:RING_CELLS + dy + h, RING_CELLS + dx:RING_CELLS + dx + w])
    return out


def persistence(frames):
    """Per-cell HUD persistence over a run of frames (0..1, CELL px cells)."""
    lumas = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames]
    grads = [_gradient(l) for l in lumas]
    p = None
    for i in range(1, len(frames)):
        (ax, ay), (bx, by) = grads[i - 1], grads[i]
        energy = bx * bx + by * by
        drift_px = (bx - ax) ** 2 + (by - ay) ** 2
        # Per pixel: an edge, and one that kept its gradient. A HUD stroke
        # thinner than a cell shares the cell with moving scene; counting
        # its steady edge pixels finds it where a whole-cell test drowns it
        # in the scene's changing edges.
        strong = energy > EDGE_LEVEL ** 2 * 4
        # ...or a pixel whose own colour stayed while it stands out from its
        # surroundings: a stroke too thin to have an inside, whose gradient
        # always takes in the moving background beside it. (Flat sky stays
        # the same too, but does not stand out.)
        same = np.abs(frames[i].astype(np.int16) - frames[i - 1].astype(np.int16)).max(axis=2) <= STATIC_LEVEL / 2
        contrast = np.abs(lumas[i] - cv2.blur(lumas[i], (5, 5))) > CONTRAST_LEVEL
        kept = (strong & (drift_px < STRUCTURE_CHANGE * energy)) | (same & contrast)
        strong = strong | contrast
        steady_count = _pool(kept.astype(np.float32), np.sum)
        strong_count = _pool(strong.astype(np.float32), np.sum)
        change = _pool(np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32)).max(axis=2), np.max)
        motion = _pool(np.abs(lumas[i] - lumas[i - 1]), np.mean)
        edges = steady_count >= STEADY_PIXELS
        steady = steady_count >= 0.25 * strong_count
        changed = (change > CHANGE_LEVEL) | (strong_count >= STEADY_PIXELS) & (steady_count == 0)
        moving = _ring_max(motion) > MOTION_LEVEL
        if p is None:
            p = np.zeros(steady_count.shape, np.float32)
        rise = edges & steady & moving
        p = np.where(rise, p + RISE * (1 - p), np.where(changed, p * FALL, p))
    return p


def suggest(frames):
    """HUD suggestion for the last frame: 255 = HUD, 0 = not, at frame size.

    Cells that reached one half of persistence, joined to their neighbours
    and with enclosed holes filled (the flat inside of a panel or bar has no
    edges of its own), then narrowed to the pixels that stayed the same over
    the run: a HUD element is fixed, the scene showing between its glyphs is
    not.
    """
    h, w = frames[-1].shape[:2]
    # Large frames are looked at at half size: a 1440p run of 24 frames took
    # 6.5 s at full size, and the HUD is just as clear at half.
    scale = 2 if w > 1600 else 1
    small = frames if scale == 1 else [cv2.resize(f, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
                                       for f in frames]
    p = persistence(small)
    cells = (p >= 0.5).astype(np.uint8)
    if scale == 1:   # half-size cells already reach 8 px; growing them took in the scene around
        cells = cv2.dilate(cells, np.ones((3, 3), np.uint8))
    cells = cv2.morphologyEx(cells, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Fill enclosed holes: flood the background from the border.
    flood = np.pad(cells, 1)
    mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, mask, (0, 0), 2)
    cells = (flood[1:-1, 1:-1] != 2).astype(np.uint8)
    region = cv2.resize(cells, (cells.shape[1] * CELL * scale, cells.shape[0] * CELL * scale),
                        interpolation=cv2.INTER_NEAREST)
    full = np.zeros((h, w), np.uint8)
    full[:region.shape[0], :region.shape[1]] = region
    stack = np.stack([f.astype(np.int16) for f in frames[-8:]])
    steady = (stack.max(axis=0) - stack.min(axis=0)).max(axis=2) <= STATIC_LEVEL
    out = (full.astype(bool) & steady).astype(np.uint8)
    # Drop specks the user would only have to erase.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(out, connectivity=8)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] < 24:
            out[labels == i] = 0
    return out * 255


# ---------------------------------------------------------------- project

def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_png(path, image):
    ok, encoded = cv2.imencode('.png', image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    data = encoded.tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + '.tmp')
    temporary.write_bytes(data)
    os.replace(temporary, path)
    return sha256_bytes(data)


def source_of(video):
    """The original recording's identity: its file name and size (a renamed
    or re-encoded copy of the same footage still has to be kept in one split
    by the user)."""
    path = Path(video)
    return f'video:{path.name}:{path.stat().st_size}'


def fit_size(width, height, target=None):
    """Label size: `target` (w, h) when given, else the frame size cropped
    down to multiples of 16 (the trainer's requirement)."""
    if target:
        return int(target[0]) // 16 * 16, int(target[1]) // 16 * 16
    return width // 16 * 16, height // 16 * 16


def prepare(image, size, interpolation):
    """Resize (when the label size differs from the frame's) or crop the
    frame's bottom/right remainder to the multiple-of-16 size."""
    w, h = size
    if image.shape[1] // 16 * 16 == w and image.shape[0] // 16 * 16 == h:
        return np.ascontiguousarray(image[:h, :w])
    return cv2.resize(image, (w, h), interpolation=interpolation)


class Project:
    """A label folder: labels.json (me-ui-labels-1) and samples/<id>/."""

    def __init__(self, folder):
        self.folder = Path(folder).resolve()
        self.path = self.folder / PROJECT_FILE
        if self.path.exists():
            self.manifest = json.loads(self.path.read_text(encoding='utf-8'))
            if self.manifest.get('format') != LABEL_FORMAT or not isinstance(self.manifest.get('samples'), list):
                raise ValueError(f'{self.path} is not a {LABEL_FORMAT} label project')
        else:
            if self.folder.exists() and any(self.folder.iterdir()):
                raise ValueError('Choose an empty folder for a new label project')
            self.manifest = {'format': LABEL_FORMAT, 'label_version': 'hud-paint-1', 'samples': []}

    # -- queries
    def samples(self):
        return self.manifest['samples']

    def sample(self, sample_id):
        return next((s for s in self.samples() if s['id'] == sample_id), None)

    def split_of(self, source):
        return next((s['split'] for s in self.samples() if s['source'] == source), None)

    def counts(self):
        counts = {split: 0 for split in SPLITS}
        for s in self.samples():
            counts[s['split']] += 1
        return counts

    def coverage(self, sample_id):
        s = self.sample(sample_id)
        if not s:
            return None
        return cv2.imread(str(self.folder / s['coverage']['path']), cv2.IMREAD_GRAYSCALE)

    def valid(self, sample_id):
        s = self.sample(sample_id)
        if not s:
            return None
        return cv2.imread(str(self.folder / s['valid']['path']), cv2.IMREAD_GRAYSCALE)

    # -- changes
    def save(self, video, last_frame, frames, coverage, valid, split, size=None):
        """Store one reviewed sample: the three frames ending at `last_frame`
        and the label painted on the last one. Saving the same frame of the
        same video again replaces it. Returns the sample id."""
        if split not in SPLITS:
            raise ValueError('split must be train, val or test')
        video = Path(video).resolve()
        source = source_of(video)
        if video.is_relative_to(LOCKED_CAPTURE.resolve()):
            split = 'test'   # the locked capture is test-only (mefi.ui_data)
        fixed = self.split_of(source)
        if fixed and fixed != split:
            raise ValueError(f'This video is already in the {fixed} split; one recording cannot be in two '
                             f'splits (it would leak into validation).')
        if len(frames) != 3:
            raise ValueError('A sample needs the three frames ending at the labeled one')
        h, w = frames[-1].shape[:2]
        size = fit_size(w, h, size)
        if size[0] < 64 or size[1] < 64:
            raise ValueError('Frames are too small to label (64 px minimum)')
        sample_id = f'{video.stem}-{last_frame:07d}'
        rel = Path('samples') / sample_id
        refs = []
        for k, frame in enumerate(frames):
            name = rel / f'frame{k}.png'
            refs.append({'path': name.as_posix(),
                         'sha256': write_png(self.folder / name, prepare(frame, size, cv2.INTER_AREA))})
        coverage = prepare(coverage, size, cv2.INTER_AREA)
        valid = (prepare(valid, size, cv2.INTER_NEAREST) >= 128).astype(np.uint8) * 255
        entry = {'id': sample_id, 'source': source, 'split': split,
                 'frame_ids': [last_frame - 2, last_frame - 1, last_frame],
                 'width': size[0], 'height': size[1], 'frames': refs,
                 'coverage': {'path': (rel / 'coverage.png').as_posix(),
                              'sha256': write_png(self.folder / rel / 'coverage.png', coverage)},
                 'valid': {'path': (rel / 'valid.png').as_posix(),
                           'sha256': write_png(self.folder / rel / 'valid.png', valid)}}
        samples = [s for s in self.samples() if s['id'] != sample_id]
        samples.append(entry)
        samples.sort(key=lambda s: s['id'])
        self.manifest['samples'] = samples
        self._write()
        return sample_id

    def remove(self, sample_id):
        self.manifest['samples'] = [s for s in self.samples() if s['id'] != sample_id]
        self._write()

    def _write(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.manifest, indent=1), encoding='utf-8')
        os.replace(temporary, self.path)
