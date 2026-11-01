"""Decode standalone offline text maps and draw NEW diagnostic copies only.

Axis-aligned contour/component ROIs, not OCR recognition, semantic HUD labels
or native masks. Optional manual references measure box coverage only; they do
not influence decoding. No activation, training, capture or model changes.
"""
import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw


def decode_maps(raw, kind, binary, score_floor):
    candidates = []
    if kind == 'db':
        scores = raw[0, 0]
        mask = (scores >= binary).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            if min(w, h) < 3 or area <= 0 or perimeter <= 0:
                continue
            local = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(local, [contour - [x, y]], 1)
            score = float(cv2.mean(scores[y:y + h, x:x + w], local)[0])
            if score < score_floor:
                continue
            # Axis-aligned offset approximation for this ROI-only experiment,
            # NOT PaddleOCR's exact rotated PyClipper unclip implementation.
            pad = area * 1.4 / perimeter
            candidates.append((x - pad, y - pad, x + w + pad, y + h + pad, score))
    else:
        scores, links = raw[0, :, :, 0], raw[0, :, :, 1]
        text = (scores >= binary).astype(np.uint8)
        link = (links >= .4).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(np.clip(text + link, 0, 1), connectivity=4)
        for index in range(1, count):
            x, y, w, h, area = map(int, stats[index])
            component = labels[y:y + h, x:x + w] == index
            score = float(scores[y:y + h, x:x + w][component].max())
            if area < 10 or score < score_floor:
                continue
            pad = int(math.sqrt(area * min(w, h) / (w * h)) * 2)
            candidates.append((x - pad, y - pad, x + w + pad, y + h + pad, score))
    return sorted(candidates, key=lambda r: -r[4])


def source_regions(candidates, map_side, size, width, height):
    input_width, input_height = (size, size) if isinstance(size, int) else size
    map_width, map_height = (map_side, map_side) if isinstance(map_side, int) else map_side
    scale = min(input_width / width, input_height / height)
    rw, rh = max(1, math.floor(width * scale + .5)), max(1, math.floor(height * scale + .5))
    px, py = math.floor((input_width - rw) / 2), math.floor((input_height - rh) / 2)
    regions = []
    for left, top, right, bottom, score in candidates:
        left, right = np.clip((np.array([left, right]) * input_width / map_width - px) * width / rw, 0, width)
        top, bottom = np.clip((np.array([top, bottom]) * input_height / map_height - py) * height / rh, 0, height)
        area = (right - left) * (bottom - top)
        if not 0 < area <= .5 * width * height:
            continue
        regions.append(dict(left=float(left), top=float(top), right=float(right), bottom=float(bottom), confidence=score))
    return regions


def coverage(regions, bounds):
    left, top, right, bottom = map(int, bounds)
    mask = np.zeros((bottom - top, right - left), dtype=bool)
    for r in regions:
        x0, y0 = max(left, math.floor(r['left'])), max(top, math.floor(r['top']))
        x1, y1 = min(right, math.ceil(r['right'])), min(bottom, math.ceil(r['bottom']))
        if x1 > x0 and y1 > y0:
            mask[y0 - top:y1 - top, x0 - left:x1 - left] = True
    return float(mask.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--images', type=Path, required=True)
    parser.add_argument('--kind', choices=('db', 'craft'), required=True)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--height', type=int, help='Input height; defaults to --size (square)')
    parser.add_argument('--binary', type=float)
    parser.add_argument('--score', type=float)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    binary = args.binary if args.binary is not None else (.2 if args.kind == 'db' else .4)
    score = args.score if args.score is not None else (.4 if args.kind == 'db' else .7)
    if not 0 < binary <= 1 or not 0 < score <= 1:
        parser.error('Thresholds must be in (0,1]')
    args.out.mkdir(parents=True, exist_ok=False)
    timing = json.loads((args.run_dir / 'timings.json').read_text())
    shape = timing['output_shape']
    height = args.height or args.size
    expected = [1, 1, height, args.size] if args.kind == 'db' else [1, height // 2, args.size // 2, 2]
    if shape != expected:
        raise ValueError('Recorded output shape does not match requested detector kind/input dimensions')
    references = json.loads(args.references.read_text()) if args.references else {}
    reports = []
    for frame in timing['frames']:
        label = frame['input']
        raw = np.fromfile(args.run_dir / (label + '.f32'), dtype=np.float32).reshape(shape)
        if not np.isfinite(raw).all():
            raise ValueError('Nonfinite probability/score map')
        map_size = (shape[3], shape[2]) if args.kind == 'db' else (shape[2], shape[1])
        input_size = (args.size, args.height or args.size)
        candidates = decode_maps(raw, args.kind, binary, score)
        original = args.images / (label + '.png')
        if original.is_file():
            image = Image.open(original).convert('RGB')
            width, height = image.size
        elif label == 'blank-control':
            image = None
            width, height = args.size, args.height or args.size
        else:
            raise FileNotFoundError(original)
        all_regions = source_regions(candidates, map_size, input_size, width, height)
        regions = all_regions[:32]
        samples = []
        for _ in range(10):
            start = time.perf_counter()
            source_regions(decode_maps(raw, args.kind, binary, score), map_size, input_size, width, height)
            samples.append((time.perf_counter() - start) * 1000)
        row = {'image': label, 'region_count_before_cap': len(all_regions), 'regions': regions,
               'manual_reference_box_union_coverage': {name: coverage(regions, b) for name, b in references.items()},
               'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
               'decode_python_wall_ms': samples}
        reports.append(row)
        if image is not None:
            draw = ImageDraw.Draw(image)
            for r in regions:
                draw.rectangle(tuple(r[k] for k in ('left', 'top', 'right', 'bottom')), outline='red', width=3)
                draw.text((r['left'], r['top']), f"{r['confidence']:.3f}", fill='yellow')
            image.save(args.out / (label + '.boxes.png'))
    report = {'kind': args.kind, 'binary_threshold': binary, 'score_threshold': score,
              'input_width': args.size, 'input_height': args.height or args.size, 'regions_cap': 32, 'activated': False,
              'decoding': 'Axis-aligned ROI approximation, not native mask or exact source rotated-polygon postprocess',
              'coverage_note': 'Manual rectangle coverage on supplied case only; NOT ground-truth text/HUD accuracy',
              'timing_note': 'Python component/contour decode measured separately; not deployed native CPU performance',
              'frames': reports}
    (args.out / 'evaluation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({r['image']: {'regions': len(r['regions']), 'coverage': r['manual_reference_box_union_coverage']}
                      for r in reports}, indent=2))


if __name__ == '__main__':
    main()
