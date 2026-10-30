"""Counterbalanced native pipeline timing; stdlib only (no PyTorch GPU context).

--inputs are folders containing a.png and b.png at the desired output size.
--models accepts ONNX paths and/or 'dis'. Original inputs/models are read only.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess


def summarize(values, budget_ms=4.0):
    if not math.isfinite(budget_ms) or budget_ms <= 0:
        raise ValueError('budget must be finite and positive')
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('latency samples must be nonempty, finite and nonnegative')
    ordered = sorted(values)
    def percentile(q):
        index = (len(ordered) - 1) * q
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)
    return {'count': len(values), 'mean_ms': statistics.mean(values), 'median_ms': percentile(0.5),
            'p95_ms': percentile(0.95), 'p99_ms': percentile(0.99), 'min_ms': ordered[0],
             'max_ms': ordered[-1], 'over_2ms': sum(v > 2 for v in values),
            'all_samples_within_2ms': ordered[-1] <= 2, 'budget_ms': budget_ms,
            'over_budget': sum(v > budget_ms for v in values),
            'all_samples_within_budget': ordered[-1] <= budget_ms, 'samples_ms': values}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--executable')
    p.add_argument('--models', nargs='+')
    p.add_argument('--inputs', nargs='+')
    p.add_argument('--from-report', help='re-evaluate saved raw samples; no GPU runs or training')
    p.add_argument('--budget-ms', type=float, default=4.0, help='maximum allowed stage/pipeline time (default: 4 ms)')
    p.add_argument('--refinement', choices=['dis', 'ai', 'both'], default='dis',
                   help='AI pyramid refinement path (default: backbone + DIS)')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--samples', type=int, default=100)
    args = p.parse_args()
    if args.rounds < 1 or not 1 <= args.samples <= 200:
        p.error('positive rounds and 1..200 samples required')
    if not math.isfinite(args.budget_ms) or args.budget_ms <= 0:
        p.error('budget-ms must be finite and positive')
    if args.from_report:
        source = Path(args.from_report).resolve()
        raw = source.read_bytes()
        report = json.loads(raw)
        for group in report['groups'].values():
            for stage, summary in group['stages'].items():
                group['stages'][stage] = summarize(summary['samples_ms'], args.budget_ms)
        report['acceptance'] = {'budget_ms': args.budget_ms, 'source_report': str(source),
                                'source_sha256': hashlib.sha256(raw).hexdigest(), 'new_gpu_runs': 0}
        root = Path(args.out).resolve()
        root.mkdir(parents=True, exist_ok=False)
        (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        for label, group in report['groups'].items():
            gpu = group['stages']['pipeline_gpu_sum']
            print(json.dumps({'label': label, **{key: gpu[key] for key in
                ('count', 'median_ms', 'p95_ms', 'max_ms', 'budget_ms', 'over_budget', 'all_samples_within_budget')}}))
        return
    if not args.executable or not args.models or not args.inputs:
        p.error('executable, models and inputs are required unless --from-report is used')
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    executable = str(Path(args.executable).resolve())
    models = [Path(m).resolve() if m != 'dis' else None for m in args.models]
    inputs = [Path(folder).resolve() for folder in args.inputs]
    assert len({m.stem if m else 'dis' for m in models}) == len(models), 'Model names must be unique'
    modes = ('dis', 'ai') if args.refinement == 'both' else (args.refinement,)
    jobs = [(model, mode) for model in models for mode in (modes if model else ('dis',))]
    env = os.environ.copy()
    env['MOTION_ENHANCER_BENCH_ITERS'] = str(args.samples)
    env['MOTION_ENHANCER_BENCH_SERIAL'] = '1'
    env.pop('MOTION_ENHANCER_AI_TEST_INPUT', None)
    env.pop('MOTION_ENHANCER_AI_TEST_OUTPUT', None)
    report = {'native_sha256': hashlib.sha256(Path(executable).read_bytes()).hexdigest(),
              'arguments': vars(args), 'order': [], 'groups': {}, 'gaming_load_tested': False}
    for round_index in range(args.rounds):
        order = jobs if round_index % 2 == 0 else jobs[::-1]
        for model, mode in order:
            name = f'{model.stem}-{mode}-refinement' if model else 'dis'
            if model:
                env['MOTION_ENHANCER_AI_FLOW_MODEL'] = str(model)
            else:
                env.pop('MOTION_ENHANCER_AI_FLOW_MODEL', None)
            for folder in inputs:
                label = f'{name}-{folder.name}'
                logfile = root / f'{label}-round-{round_index}.log'
                command = [executable, '--gpu-offline', str(folder / 'a.png'), str(folder / 'b.png'),
                           str(root / f'{label}.png'), '--flow-engine', 'ai' if model else 'dis',
                           '--no-ui-mask', '--adapter', 'dgpu']
                if model and mode == 'ai':
                    command.append('--ai-refinement')
                result = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=180)
                log = result.stdout.decode('utf-8', errors='replace') + result.stderr.decode('utf-8', errors='replace')
                logfile.write_text(log, encoding='utf-8')
                assert result.returncode == 0, log[-3000:]
                if model:
                    assert 'Flow estimator: AI pyramid flow (ONNX)' in log, 'Silent fallback'
                    assert ('AI refinement selected: integrated AI; DIS patch passes: 0' if mode == 'ai'
                            else 'AI refinement selected: DIS;') in log, 'Wrong refinement mode'
                metrics = [json.loads(line[len('BENCH_JSON '):]) for line in log.splitlines()
                           if line.startswith('BENCH_JSON ')]
                assert len(metrics) == 4 and all(row['count'] == args.samples for row in metrics)
                report['order'].append({'round': round_index, 'model': name, 'refinement': mode, 'input': str(folder)})
                group = report['groups'].setdefault(label, {'model': str(model) if model else 'dis',
                    'input': str(folder), 'refinement': mode, 'stages': {}})
                for row in metrics:
                    previous = group['stages'].get(row['stage'], {}).get('samples_ms', [])
                    group['stages'][row['stage']] = summarize(previous + row['samples_ms'], args.budget_ms)
                (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                gpu = group['stages']['pipeline_gpu_sum']
                print(json.dumps({'model': name, 'input': folder.name, 'rounds': round_index + 1,
                    **{key: gpu[key] for key in ('count', 'mean_ms', 'median_ms', 'p95_ms', 'max_ms',
                                                 'budget_ms', 'over_budget', 'all_samples_within_budget')}}), flush=True)


if __name__ == '__main__':
    main()
