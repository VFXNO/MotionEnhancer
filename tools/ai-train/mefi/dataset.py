"""Triplet dataset builder: (frame0, frame1, frame2) from your own videos.

frame1 is the true middle of frame0 and frame2, so training needs no motion
labels: the model predicts motion between frame0 and frame2 and is scored
on how well that motion rebuilds frame1.

    python -m mefi.dataset --out datasets/mygame --height 360 video1.mp4 video2.mkv

Per video, frames are decoded by ffmpeg already scaled to --height (the
network works at a quarter of the capture, so 360 matches 1440p capture),
and a triplet is kept only when it is a clean sample of motion:

  * no duplicated frame inside it (recorded games often repeat frames when
    the game ran below the recording rate; a repeat breaks "frame1 is the
    middle")
  * no scene cut (frame0 and frame2 differ too much, or one half of the
    triplet changes far more than the other)
  * motion between frame0 and frame2 within --min-motion .. --max-motion
    pixels (median, by OpenCV's DIS flow at a small size)
  * at least --gap frames after the previous kept triplet, at most
    --max-per-video triplets

Each triplet is written as im0.jpg, im1.jpg, im2.jpg in its own folder, and
index.json lists them with their video, frame and motion. Building into an
existing dataset adds to it.

Progress goes to stdout as JSON lines (the trainer app reads them).
"""
import argparse
import json
import os
import subprocess
import sys
import time

import cv2
import numpy as np


def emit(**event):
    print(json.dumps(event), flush=True)


def probe(path):
    """Displayed width and height, frame rate and frame count (ffprobe).

    The displayed size, not the stored one: a phone video is often stored
    landscape with a rotation flag, and ffmpeg outputs it rotated, so its
    stored size squashed every portrait frame into a landscape one; and a
    non-square pixel aspect (some DVD/broadcast files) stretches it."""
    out = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_packets',
         '-show_entries', 'stream=width,height,r_frame_rate,nb_read_packets,sample_aspect_ratio'
         ':stream_side_data=rotation:stream_tags=rotate', '-of', 'json', path],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)['streams'][0]
    num, den = s['r_frame_rate'].split('/')
    fps = float(num) / max(float(den), 1.0)
    w, h = int(s['width']), int(s['height'])
    sar = s.get('sample_aspect_ratio', '1:1')
    try:
        a, b = (int(v) for v in sar.split(':'))
        if a > 0 and b > 0:
            w = int(round(w * a / b))
    except ValueError:
        pass
    rotation = 0.0
    for side in s.get('side_data_list', []):
        if 'rotation' in side:
            rotation = float(side['rotation'])
    if not rotation and 'rotate' in s.get('tags', {}):
        rotation = float(s['tags']['rotate'])
    if int(round(abs(rotation))) % 180 == 90:
        w, h = h, w
    return w, h, fps, int(s.get('nb_read_packets', 0) or 0)


def frames(path, height):
    """Yields RGB uint8 frames scaled to `height` (width even, aspect kept)."""
    w0, h0, _, _ = probe(path)
    width = int(round(w0 * height / h0 / 2)) * 2
    proc = subprocess.Popen(
        ['ffmpeg', '-v', 'error', '-i', path, '-vf', f'scale={width}:{height}:flags=area',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=width * height * 3 * 4)
    size = width * height * 3
    try:
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            yield np.frombuffer(buf, np.uint8).reshape(height, width, 3)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def mean_abs_diff(a, b):
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


class MotionMeter:
    """Median motion (pixels at the dataset scale) between two frames."""

    def __init__(self):
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

    def __call__(self, a, b, factor=4):
        ga = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY)
        gb = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY)
        small = (max(16, a.shape[1] // factor), max(16, a.shape[0] // factor))
        ga = cv2.resize(ga, small, interpolation=cv2.INTER_AREA)
        gb = cv2.resize(gb, small, interpolation=cv2.INTER_AREA)
        flow = self.dis.calc(ga, gb, None)
        mag = np.sqrt((flow ** 2).sum(-1)) * factor
        return float(np.median(mag)), float(np.percentile(mag, 90))


def build(args):
    os.makedirs(args.out, exist_ok=True)
    index_path = os.path.join(args.out, 'index.json')
    index = []
    if os.path.exists(index_path):
        with open(index_path, encoding='utf-8') as f:
            index = json.load(f)
    # Entries whose images are gone (a deleted `triplets` folder) are
    # forgotten, so this build makes them again instead of skipping them as
    # already made.
    index = [e for e in index
             if all(os.path.exists(os.path.join(args.out, 'triplets', e['dir'], f'im{i}.jpg')) for i in range(3))]
    known = {e['dir'] for e in index}
    meter = MotionMeter()
    strides = sorted(set(args.strides))
    reach = 2 * strides[-1]
    started = time.time()
    total_kept = 0
    for v, video in enumerate(args.videos):
        stem = os.path.splitext(os.path.basename(video))[0]
        stem = ''.join(c if c.isalnum() or c in '-_' else '_' for c in stem)[:48]
        try:
            _, _, fps, count = probe(video)
        except Exception as e:  # noqa: BLE001 - reported to the app
            emit(event='error', video=video, message=f'cannot read: {e}')
            continue
        # The video's triplets spread over all of it: with the plain gap the
        # limit per video was reached early in a long one (600 triplets every
        # 6 frames took the first 4065 frames, about a minute, of an 11-minute
        # gameplay video) and the rest was never used.
        gap = max(args.gap, count // max(1, args.max_per_video)) if count else args.gap
        emit(event='video', index=v, videos=len(args.videos), video=video, frames=count, fps=fps, gap=gap)
        history = []            # the last `reach` + 1 frames
        kept = 0
        taken = 0               # kept now or already in the dataset
        last_kept = -10 ** 9
        rejected = {'duplicate': 0, 'cut': 0, 'still': 0, 'fast': 0}
        for i, frame in enumerate(frames(video, args.height)):
            history.append(frame)
            if len(history) > reach + 1:
                history.pop(0)
            if i % 30 == 0:
                emit(event='progress', index=v, frame=i, frames=count, kept=kept, rejected=rejected)
            if taken >= args.max_per_video:
                break
            if i - last_kept < gap:
                continue
            for s in strides:
                if len(history) < 2 * s + 1:
                    continue
                f0, f1, f2 = history[-1 - 2 * s], history[-1 - s], history[-1]
                d01, d12, d02 = mean_abs_diff(f0, f1), mean_abs_diff(f1, f2), mean_abs_diff(f0, f2)
                if min(d01, d12) < args.duplicate:
                    rejected['duplicate'] += 1
                    continue
                if d02 > args.cut or max(d01, d12) > args.asymmetry * max(min(d01, d12), 0.5):
                    rejected['cut'] += 1
                    continue
                median, p90 = meter(f0, f2)
                if p90 < args.min_motion:
                    rejected['still'] += 1
                    continue
                if median > args.max_motion:
                    rejected['fast'] += 1
                    continue
                name = f'{stem}_{i - 2 * s:06d}_s{s}'
                folder = os.path.join(args.out, 'triplets', name)
                if name in known:
                    # Made by an earlier build: the slot is taken, as it was
                    # then. Skipping it to try the next stride shifted the
                    # sampling and added near-copies of the clips already
                    # there (73 of them when one video was built twice).
                    taken += 1
                    last_kept = i
                    break
                os.makedirs(folder, exist_ok=True)
                for k, img in enumerate((f0, f1, f2)):
                    cv2.imwrite(os.path.join(folder, f'im{k}.jpg'), cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                                [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                entry = {'dir': name, 'video': stem, 'frame': i - 2 * s, 'stride': s,
                         'motion': round(median, 2), 'motion90': round(p90, 2),
                         'width': int(f0.shape[1]), 'height': int(f0.shape[0])}
                index.append(entry)
                known.add(name)
                kept += 1
                taken += 1
                total_kept += 1
                last_kept = i
                emit(event='sample', path=folder, motion=entry['motion'], kept=kept)
                break
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=1)
        emit(event='video_done', index=v, video=video, kept=kept, rejected=rejected)
    emit(event='done', kept=total_kept, total=len(index), seconds=round(time.time() - started, 1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('videos', nargs='+')
    p.add_argument('--out', required=True, help='dataset folder (created or added to)')
    p.add_argument('--height', type=int, default=360, help='frame height in the dataset (default 360)')
    p.add_argument('--strides', type=int, nargs='+', default=[1, 2],
                   help='frame gaps to try: 1 = consecutive frames, 2 = every other frame (larger motion)')
    p.add_argument('--max-per-video', type=int, default=600)
    p.add_argument('--gap', type=int, default=6, help='frames between kept triplets')
    p.add_argument('--min-motion', type=float, default=1.0, help='px (90th percentile) - below is a still')
    p.add_argument('--max-motion', type=float, default=40.0, help='px (median) - above is too fast')
    p.add_argument('--duplicate', type=float, default=0.6, help='mean abs difference (levels) of a repeated frame')
    p.add_argument('--cut', type=float, default=45.0, help='mean abs difference (levels) of a scene cut')
    p.add_argument('--asymmetry', type=float, default=4.0, help='one half changing this much more = a cut')
    p.add_argument('--quality', type=int, default=95, help='JPEG quality')
    args = p.parse_args(argv)
    build(args)


if __name__ == '__main__':
    main()
