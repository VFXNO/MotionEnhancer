"""Opt-in standalone DirectML experiment on export_probe's synthetic fixtures.

No training or captures. Original/feature-tap/temporal graphs run in opposite
orders; placement runs are separate. An untrained fixture cannot prove accuracy,
trained-checkpoint parity, a live queue contract, or a gaming-load deadline.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np

from export_probe import new_output


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def statistics(values):
    values = np.asarray(values, dtype=np.float64)
    return {'count': len(values), 'mean_ms': float(values.mean()), 'p50_ms': float(np.median(values)),
            'p95_ms': float(np.percentile(values, 95)), 'p99_ms': float(np.percentile(values, 99)),
            'max_ms': float(values.max()), 'above_4ms': int((values > 4).sum())}


def placement(folder):
    files = list(folder.glob('placement*.json'))
    if len(files) != 1:
        raise ValueError('Need one actual ORT placement trace')
    events = json.loads(files[0].read_text(encoding='utf-8'))
    counts = Counter(e.get('args', {}).get('provider') for e in events if e.get('args', {}).get('provider'))
    if not counts or set(counts) != {'DmlExecutionProvider'}:
        raise ValueError(f'GPU-only placement not established: {dict(counts)}')
    return {'profile': str(files[0]), 'profile_sha256': sha256(files[0]), 'provider_events': dict(counts)}


def command(args, model, manifest, output, runs, profile):
    result = subprocess.run([str(args.probe.resolve()), str(args.runtime.resolve()), str(model.resolve()),
                             str(manifest.resolve()), str(output.resolve()), str(runs), str(int(profile))],
                            capture_output=True, timeout=180)
    log = (result.stdout + result.stderr).decode('utf-8', errors='replace')
    (output.parent / f'{output.name}.log').write_text(log, encoding='utf-8')
    if result.returncode:
        raise RuntimeError(f'{output.name} failed: {log[-5000:]}')
    timing = json.loads((output / 'timings.json').read_text())
    if not timing.get('completed') or timing.get('cpu_fallback') is not False:
        raise ValueError('Incomplete or CPU-fallback run')
    return timing


def array(path):
    value = np.fromfile(path, dtype=np.float32)
    if not len(value) or not np.isfinite(value).all():
        raise ValueError(f'Empty/nonfinite output: {path}')
    return value


def compare(left, right):
    if left.shape != right.shape:
        raise ValueError('Output comparison shape mismatch')
    delta = np.abs(left - right)
    return {'exact': bool(np.array_equal(left, right)), 'max_abs': float(delta.max()),
            'p99_abs': float(np.percentile(delta, 99))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=24)
    args = parser.parse_args(argv)
    if not 4 <= args.runs <= 500:
        parser.error('--runs must be 4..500')
    prepared = args.prepared.resolve()
    metadata = json.loads((prepared / 'preparation.json').read_text())
    if not metadata.get('synthetic_inputs_only') or not metadata.get('untrained_fixture') or metadata.get('deployable') is not False:
        raise ValueError('Only explicit synthetic research preparations are accepted')
    models = {r['variant']: r for r in metadata['variants']}
    if set(models) != {'baseline', 'tap', 'temporal'}:
        raise ValueError('Need exactly baseline, tap and temporal variants')
    for name, record in models.items():
        if sha256(prepared / f'{name}.onnx') != record['sha256']:
            raise ValueError('Prepared model SHA256 changed')
    out = new_output(args.out)
    protected = {str(p.resolve()): sha256(p) for p in [args.probe, args.runtime / 'onnxruntime.dll',
                 args.runtime / 'DirectML.dll', prepared / 'research-weights.pt',
                 *[prepared / f'{n}.onnx' for n in models]]}
    report = {'status': 'in_progress', 'synthetic_only': True, 'optimizer_steps_run': 0,
              'gaming_load_tested': False, 'accuracy_established': False,
              'trained_checkpoint_parity_established': False, 'latency_guarantee_established': False,
              'tolerances': {'flow_max_abs_network_pixels': .01, 'blend_max_abs': .001},
              'python': sys.version, 'platform': platform.platform(), 'protected_before': protected,
              'placement': {}, 'rounds': [], 'checks': {}}

    def save():
        (out / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    save()
    try:
        # Separate these traces from all cost samples.
        for name in ('baseline', 'tap', 'temporal'):
            folder = out / f'placement-{name}'
            command(args, prepared / f'{name}.onnx', prepared / f'{name}-inputs.tsv', folder, 4, True)
            report['placement'][name] = placement(folder)
            save()
        for order in (('baseline', 'tap', 'temporal'), ('temporal', 'tap', 'baseline')):
            round_index = len(report['rounds'])
            results = {}
            for name in order:
                folder = out / f'round{round_index}-{name}'
                timing = command(args, prepared / f'{name}.onnx', prepared / f'{name}-inputs.tsv',
                                 folder, args.runs, False)
                repeat_drift = max(c['repeat_max_abs_drift'] for c in timing['cases'])
                if repeat_drift != 0 or timing['first_input_restore_max_abs_drift'] != 0:
                    raise ValueError(f'{name}: fixed-input repeat/restore was not deterministic')
                wall = [r['wall_ms'] for case in timing['cases'] for r in case['samples']]
                gpu = [r['gpu_ms'] for case in timing['cases'] for r in case['samples']]
                results[name] = {'wall': statistics(wall), 'gpu_timeline': statistics(gpu),
                                 'scope': timing['scope'], 'first_input_restore_max_abs_drift': timing['first_input_restore_max_abs_drift'],
                                 'repeat_max_abs_drift': repeat_drift}
            checks = {}
            for case in ('startup', 'history_a', 'history_b', 'changed'):
                checks[case] = {}
                for name in ('tap', 'temporal'):
                    checks[case][name] = {}
                    for key, tolerance in (('flow', .01), ('mask', .001)):
                        expected = array(out / f'round{round_index}-baseline' / f'{case}-{key}.f32')
                        actual = array(out / f'round{round_index}-{name}' / f'{case}-{key}.f32')
                        metric = compare(expected, actual)
                        checks[case][name][key] = metric
                        if metric['max_abs'] > tolerance:
                            raise ValueError(f'{name} changed {key} beyond the predeclared synthetic tolerance')
                for name, record in models.items():
                    for key in record['outputs']:
                        reference = array(prepared / f'reference-{name}-{case}-{key}.f32')
                        actual = array(out / f'round{round_index}-{name}' / f'{case}-{key}.f32')
                        # Record eager-vs-DML differences, do not confuse them with
                        # graph-to-graph flow parity or a trained accuracy score.
                        checks[case].setdefault('eager_vs_dml', {})[f'{name}-{key}'] = compare(reference, actual)
            folder = out / f'round{round_index}-temporal'
            startup = array(folder / 'startup-ui_allow.f32')
            if np.count_nonzero(startup):
                raise ValueError('Invalid startup history authorized UI pixels')
            a, b = array(folder / 'history_a-ui_allow.f32'), array(folder / 'history_b-ui_allow.f32')
            if np.array_equal(a, b):
                raise ValueError('Changed valid history was not consumed')
            checks['history_consumption_max_abs_difference'] = float(np.abs(a - b).max())
            checks['startup_authorized_pixels'] = 0
            report['rounds'].append({'order': order, 'results': results, 'checks': checks})
            save()
        for path, expected in protected.items():
            if sha256(path) != expected:
                raise ValueError(f'Protected probe/runtime/model file changed: {path}')
        report['protected_after_match'] = True
        report['status'] = 'completed_synthetic_feasibility_only'
        save()
    except BaseException as error:
        report['status'] = 'failed'
        report['error'] = str(error)
        save()
        raise
    print(json.dumps({'report': str(out / 'report.json'), 'status': report['status'],
                      'rounds': [{name: r['results'][name]['gpu_timeline'] for name in r['order']}
                                 for r in report['rounds']]}, indent=2))


if __name__ == '__main__':
    main()
