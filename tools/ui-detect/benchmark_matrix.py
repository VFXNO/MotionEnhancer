"""Compare unchanged DirectML FP16 with explicit, ISA-verified UI matrix inference.

No training or activation. New output directory only. Includes the matrix
backend's D3D12 producer-fence wait and shared GPU input in worker wall times.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np
from benchmark_int8 import agreement, stats


def raw_agreement(reference, candidate):
    a = np.fromfile(reference, dtype=np.float32).reshape(5, 8400)
    b = np.fromfile(candidate, dtype=np.float32).reshape(5, 8400)
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise RuntimeError('Nonfinite detector proposals')
    delta = np.abs(a - b)
    relevant = np.maximum(a[4], b[4]) >= 0.10
    return {'score_mean_absolute_delta': float(delta[4].mean()),
            'score_max_absolute_delta': float(delta[4].max()),
            'relevant_proposals': int(relevant.sum()),
            'relevant_box_coordinate_median_delta_detector_px': float(np.median(delta[:4, relevant])) if relevant.any() else None,
            'relevant_box_coordinate_max_delta_detector_px': float(delta[:4, relevant].max()) if relevant.any() else None,
            'note': 'Same-input numeric agreement, not game-HUD ground-truth accuracy.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executable', type=Path, required=True)
    parser.add_argument('--fp16', type=Path, required=True)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=127)
    parser.add_argument('images', type=Path, nargs='+')
    args = parser.parse_args()
    if not 4 <= args.runs <= 500:
        parser.error('--runs must be 4..500')
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = [p.resolve() for p in args.images]
    files = [args.fp16.resolve(), args.matrix.resolve(), args.matrix.resolve().with_suffix('.bin')]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}

    def run(label, backend, image, diagnostic=False):
        env = dict(os.environ, MOTION_ENHANCER_UI_BACKEND=backend,
                   MOTION_ENHANCER_UI_MODEL=str(args.fp16.resolve()),
                   MOTION_ENHANCER_UI_MATRIX_MODEL=str(args.matrix.resolve()),
                   MOTION_ENHANCER_UI_TEST_RUNS=str(4 if diagnostic else args.runs))
        for key in ('MOTION_ENHANCER_UI_PROFILE', 'MOTION_ENHANCER_UI_MATRIX_TRACE',
                    'MOTION_ENHANCER_UI_RAW_OUTPUT', 'MOTION_ENHANCER_D3D_DEBUG',
                    'MOTION_ENHANCER_UI_MATRIX_TIMING', 'MOTION_ENHANCER_UI_TEST_ALTERNATE',
                    'MOTION_ENHANCER_UI_TEST_CHECKSUM'):
            env.pop(key, None)
        if diagnostic:
            env['MOTION_ENHANCER_UI_RAW_OUTPUT'] = str((args.out / (label + '.f32')).resolve())
            if backend == 'matrix':
                env['MOTION_ENHANCER_UI_MATRIX_TRACE'] = str((args.out / (label + '-isa')).resolve())
        output = (args.out / (label + '.json')).resolve()
        result = subprocess.run([str(args.executable.resolve()), '--ui-detect-test', str(image), str(output),
                                 '--adapter', 'dgpu'], env=env, capture_output=True, timeout=120)
        text = (result.stdout + result.stderr).decode('utf-8', errors='replace')
        (args.out / (label + '.log')).write_text(text, encoding='utf-8')
        if result.returncode:
            raise RuntimeError(f'{label} failed; see its log. No activation performed.')
        data = json.loads(output.read_text())
        if data['cpu_fallback'] is not False:
            raise RuntimeError('CPU fallback must be forbidden')
        if backend == 'matrix' and (not data['cooperative_matrix_dispatches'] or
                data['verified_wmma_dispatches'] != data['cooperative_matrix_dispatches']):
            raise RuntimeError('Expected actual dispatches with both cooperative SPIR-V and AMD v_wmma ISA')
        return data

    numeric = {}
    for index, image in enumerate(inputs):
        for backend in ('directml', 'matrix'):
            run(f'raw-{backend}-{index}', backend, image, diagnostic=True)
        numeric[str(image)] = raw_agreement(args.out / f'raw-directml-{index}.f32', args.out / f'raw-matrix-{index}.f32')
    timings = {key: {str(p): [] for p in inputs} for key in ('directml', 'matrix')}
    boxes = {key: {} for key in timings}
    rounds = []
    for round_id, order in enumerate((('directml', 'matrix'), ('matrix', 'directml'))):
        for backend in order:
            for index, image in enumerate(inputs):
                data = run(f'round{round_id}-{backend}-{index}', backend, image)
                timings[backend][str(image)].extend(data['worker_ms'])
                boxes[backend][str(image)] = data['regions']
                rounds.append({'round': round_id, 'backend': backend, 'input': str(image),
                               'timings': stats(data['worker_ms']),
                               'cold_worker_ms': data['all_worker_ms_including_cold'][0],
                               'verified_wmma_dispatches': data['verified_wmma_dispatches']})
    per_image = {}
    for image in inputs:
        key = str(image)
        box_comparison = agreement(boxes['directml'][key], boxes['matrix'][key])
        box_comparison['matrix_count'] = box_comparison.pop('int8_count')
        per_image[key] = {backend: stats(timings[backend][key]) for backend in timings}
        per_image[key]['boxes'] = box_comparison
        per_image[key]['raw'] = numeric[key]
    assert hashes == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, 'A tested model changed'
    report = {'per_image': per_image, 'rounds': rounds, 'protected_model_sha256': hashes,
              'matrix_instructions_verified': True, 'cpu_model_fallback': False, 'activated': False,
              'controlled_gaming_load': False,
               'timing_scope': 'worker including D3D12 producer wait, shared GPU input, GPU inference, proposal readback and CPU box decoding; excludes capture/flow-queue preprocessing/presentation'}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(per_image, indent=2))


if __name__ == '__main__':
    main()
