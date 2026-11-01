"""Counterbalanced native FP16/INT8 UI comparison; no capture, training or activation.

Outputs must be a NEW directory. ORT profiles check execution-provider placement,
NOT the AMD driver's exact ISA or matrix-unit utilisation. Timed runs exclude
profiling, session creation and flow-queue input preprocessing.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def stats(values):
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('Expected nonempty, positive, finite timing samples')
    values = sorted(values)
    def percentile(q):
        position = q * (len(values) - 1)
        low = int(position)
        high = min(low + 1, len(values) - 1)
        return values[low] * (high - position) + values[high] * (position - low) if high != low else values[low]
    return {'samples': len(values), 'mean_ms': statistics.mean(values), 'p50_ms': percentile(0.5),
            'p95_ms': percentile(0.95), 'p99_ms': percentile(0.99), 'max_ms': max(values),
            'over_4ms': sum(v > 4 for v in values)}


def iou(a, b):
    intersection = max(0, min(a['right'], b['right']) - max(a['left'], b['left'])) * \
                   max(0, min(a['bottom'], b['bottom']) - max(a['top'], b['top']))
    area = lambda r: (r['right'] - r['left']) * (r['bottom'] - r['top'])
    union = area(a) + area(b) - intersection
    return intersection / union if union else 0.0


def agreement(reference, candidate):
    pairs = sorted(((iou(a, b), i, j) for i, a in enumerate(reference) for j, b in enumerate(candidate)), reverse=True)
    used_a, used_b, matches = set(), set(), []
    for overlap, a, b in pairs:
        if overlap < 0.5:
            break
        if a not in used_a and b not in used_b:
            used_a.add(a)
            used_b.add(b)
            matches.append(overlap)
    return {'fp16_count': len(reference), 'int8_count': len(candidate), 'matched_iou_ge_0_5': len(matches),
            'matched_median_iou': statistics.median(matches) if matches else None,
            'note': 'Baseline agreement only, NOT ground-truth HUD accuracy. Both empty does not validate either model.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--fp16', type=Path, required=True)
    parser.add_argument('--int8', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=127, help='Runs per model/image/round, including three warmups.')
    parser.add_argument('images', type=Path, nargs='+')
    args = parser.parse_args()
    if not 4 <= args.runs <= 500:
        parser.error('--runs must be 4..500')
    args.out.mkdir(parents=True, exist_ok=False)
    models = {'fp16': args.fp16.resolve(), 'int8': args.int8.resolve()}
    hashes = {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in models.items()}
    exe = args.executable.resolve()

    def run(label, key, image, profile=False):
        env = dict(os.environ, MOTION_ENHANCER_UI_MODEL=str(models[key]),
                   MOTION_ENHANCER_UI_TEST_RUNS=str(8 if profile else args.runs))
        env.pop('MOTION_ENHANCER_UI_PROFILE', None)
        env.pop('MOTION_ENHANCER_D3D_DEBUG', None)
        if profile:
            env['MOTION_ENHANCER_UI_PROFILE'] = str((args.out / label).resolve())
        result_path = (args.out / (label + '.json')).resolve()
        result = subprocess.run([str(exe), '--ui-detect-test', str(image.resolve()), str(result_path), '--adapter', 'dgpu'],
                                env=env, capture_output=True, text=True, timeout=120)
        log = result.stdout + result.stderr
        (args.out / (label + '.log')).write_text(log, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'Native {key} test failed; see {label}.log. No activation performed.')
        data = json.loads(result_path.read_text())
        if data.get('cpu_fallback') is not False:
            raise RuntimeError('Expected the native build with CPU fallback explicitly disabled')
        if not data['worker_ms'] or any(not math.isfinite(v) or v <= 0 for v in data['worker_ms']):
            raise RuntimeError('Invalid native timing samples')
        return data, log

    placement = {}
    for key in models:
        _, log = run('profile-' + key, key, args.images[0], profile=True)
        names = [line[len('UI_PROFILE '):] for line in log.splitlines() if line.startswith('UI_PROFILE ')]
        if len(names) != 1:
            raise RuntimeError('Native ORT profile missing')
        events = json.loads(Path(names[0]).read_text())
        providers = Counter(e.get('args', {}).get('provider') for e in events if e.get('args', {}).get('provider'))
        if providers.get('CPUExecutionProvider') or not providers.get('DmlExecutionProvider'):
            raise RuntimeError(f'Expected GPU-only DML execution; got {dict(providers)}')
        placement[key] = {'providers': dict(providers), 'trace': names[0],
                          'matrix_instructions_verified': False}

    timings = {key: {str(p.resolve()): [] for p in args.images} for key in models}
    detections = {key: {} for key in models}
    rounds = []
    for round_index, order in enumerate((('fp16', 'int8'), ('int8', 'fp16'))):
        for key in order:
            for index, image in enumerate(args.images):
                name = f'round{round_index}-{key}-{index:02d}'
                data, _ = run(name, key, image)
                path = str(image.resolve())
                timings[key][path].extend(data['worker_ms'])
                detections[key][path] = data['regions']
                rounds.append({'round': round_index, 'model': key, 'input': path, 'timing': stats(data['worker_ms'])})
    per_image = {}
    for image in args.images:
        path = str(image.resolve())
        per_image[path] = {key: stats(timings[key][path]) for key in models}
        per_image[path]['agreement'] = agreement(detections['fp16'][path], detections['int8'][path])
    for key, path in models.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[key]:
            raise RuntimeError('A model changed during the benchmark')
    report = {'model_sha256': hashes, 'placement': placement, 'per_image': per_image, 'rounds': rounds,
              'controlled_gaming_load': False, 'activated': False, 'matrix_instructions_verified': False,
              'timing_scope': 'native worker wall time; no profiling, initialization, input preprocessing or presentation'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(per_image, indent=2))
    print('GPU-only INT8 is checked; matrix instruction selection still requires driver/ISA profiling. No model activated.')


if __name__ == '__main__':
    main()
