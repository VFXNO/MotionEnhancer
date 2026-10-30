"""Opt-in ROCm continuation and native DirectML numerical checks.

Run only after benchmark_refiner.py, with the rebuilt native executable.
All output is isolated; no packages, teachers, datasets or runs are installed.
"""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch

import cv2
import numpy as np
import torch

import test_quality as quality
from mefi import train
from mefi.model import Exported, load_checkpoint


@torch.inference_mode()
def known_motion(exported, image, w, h):
    image = cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2RGB)
    rows = []
    for dx, dy in ((0, 0), (4, 0), (8, 2), (16, 6), (32, 0), (64, 16)):
        shift = lambda x, y: cv2.warpAffine(image, np.array([[1, 0, x], [0, 1, y]], np.float32),
            (w, h), borderMode=cv2.BORDER_REFLECT_101)
        tensor = lambda im: torch.from_numpy(im.transpose(2, 0, 1).copy()).float().div(255).unsqueeze(0).cuda()
        a, b, target = tensor(image), tensor(shift(dx, dy)), tensor(shift(dx / 2, dy / 2))
        flow, mask = exported(torch.cat((a, b), 1))
        predicted = train.middle(a, b, flow, mask).clamp(0, 1)
        margin = max(abs(dx), abs(dy)) + 4
        region = (slice(None), slice(None), slice(margin, -margin), slice(margin, -margin))
        expected = torch.tensor([dx, dy, -dx, -dy], device='cuda').reshape(1, 4, 1, 1)
        error = (flow - expected)[region]
        epe = torch.sqrt(error[:, :2].square().sum(1))
        psnr = lambda x: float(-10 * torch.log10((x[region] - target[region]).square().mean().clamp_min(1e-10)))
        rows.append({'motion': [dx, dy], 'forward_mean_epe_pixels': float(epe.mean()),
                     'forward_median_epe_pixels': float(epe.median()), 'psnr': psnr(predicted),
                     'blend_psnr': psnr((a + b) * 0.5)})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True)
    p.add_argument('--benchmark', required=True)
    p.add_argument('--executable', required=True)
    args = p.parse_args()
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    benchmark = Path(args.benchmark).resolve()
    executable = str(Path(args.executable).resolve())
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    assert torch.cuda.is_available()
    # The fixture contains only generated triplet images, not motion labels.
    fixture = quality.TrainingTests(methodName='runTest')
    with patch.object(tempfile, 'tempdir', str(root)):
        fixture.setUp()
    report = {'gpu': torch.cuda.get_device_name(0), 'torch': str(torch.__version__), 'numerical': []}
    try:
        options = fixture.args + ['--refiner', '--epochs', '2', '--batch', '4', '--workers', '0']

        def run(arguments, name):
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                result = train.main(arguments)
            (root / f'{name}.log').write_text(out.getvalue(), encoding='utf-8')
            assert result == 0, out.getvalue()

        full = root / 'full'
        run(options + ['--run', str(full)], 'uninterrupted')
        expected = torch.load(full / 'last.pt', map_location='cpu')
        partial = root / 'partial'
        fixture.run = partial
        with fixture.stop_after(2):
            run(options + ['--run', str(partial)], 'stopped')
        paused = torch.load(partial / 'last.pt', map_location='cpu')
        assert paused['step'] == 2
        # No --refiner required: continuation must restore the architecture.
        run(['--run', str(partial), '--resume', '--epochs', '2', '--workers', '0'], 'resumed')
        actual = torch.load(partial / 'last.pt', map_location='cpu')
        parity = {}
        for key in ('model', 'opt', 'sched', 'scaler', 'rng', 'schedule'):
            fixture.assert_state_equal(expected[key], actual[key])
            parity[key] = True
        report['continuation'] = {'stopped_step': 2, 'final_step': actual['step'], 'exact': parity,
                                  'scaler': actual['scaler']}
        print(json.dumps({'event': 'gpu_resume_pass', **report['continuation']}), flush=True)
    finally:
        fixture.doCleanups()

    records = json.loads((benchmark / 'report.json').read_text())
    env = os.environ.copy()
    env.pop('MOTION_ENHANCER_BENCH_SERIAL', None)
    env.pop('MOTION_ENHANCER_BENCH_ITERS', None)
    for entry in records['exports']:
        name = entry['name']
        w, h = entry['size']
        model = benchmark / f'{name}-{w}x{h}.onnx'
        _, net = load_checkpoint(benchmark / f'{name}.pt')
        images = [cv2.imread(str(benchmark / 'cases/0' / filename)) for filename in ('a.png', 'b.png')]
        arrays = [cv2.cvtColor(cv2.resize(image, (w, h)), cv2.COLOR_BGR2RGB)
                  .astype(np.float32).transpose(2, 0, 1) / 255 for image in images]
        frames = np.concatenate(arrays, 0)[None].copy()
        input_path = root / f'{model.stem}.frames.f32'
        frames.tofile(input_path)
        prefix = root / model.stem
        env['MOTION_ENHANCER_AI_TEST_INPUT'] = str(input_path)
        env['MOTION_ENHANCER_AI_TEST_OUTPUT'] = str(prefix)
        done = subprocess.run([executable, '--ai-flow-test', str(model), '--adapter', 'dgpu'],
                              cwd=root, env=env, capture_output=True, timeout=120)
        log = done.stdout.decode('utf-8', errors='replace') + done.stderr.decode('utf-8', errors='replace')
        (root / f'{model.stem}.log').write_text(log, encoding='utf-8')
        assert done.returncode == 0 and 'output finite and mask range: PASS' in log, log[-3000:]
        with torch.inference_mode():
            exported = Exported(net.cuda().eval())
            reference = exported(torch.from_numpy(frames).cuda())
            expected = [tensor.float().cpu().numpy() for tensor in reference]
        values = [np.fromfile(str(prefix) + f'.{kind}.f32', dtype=np.float32).reshape(1, channels, h, w)
                  for kind, channels in (('flow', 4), ('mask', 1))]
        errors = [np.abs(a - b) for a, b in zip(expected, values)]
        row = {'name': name, 'size': [w, h], 'flow_mae_pixels': float(errors[0].mean()),
               'flow_p99_pixels': float(np.percentile(errors[0], 99)), 'flow_max_pixels': float(errors[0].max()),
               'mask_mae': float(errors[1].mean()), 'mask_max': float(errors[1].max())}
        # Cross-provider FP16 tolerance, not exact numerical identity.
        row['passed'] = row['flow_mae_pixels'] <= 0.1 and row['flow_p99_pixels'] <= 0.5 and row['mask_mae'] <= 0.005
        row['known_motion'] = known_motion(exported, images[0], w, h)
        report['numerical'].append(row)
        (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(row), flush=True)
        assert row['passed'], 'DirectML/PyTorch error exceeds the declared cross-provider FP16 tolerance'
        del net, reference, exported
        torch.cuda.empty_cache()
    # Real held-out application outputs, not just ONNX load tests. These
    # diagnostic subsets do not establish generalisation of short controls.
    report['native_quality'] = []
    case_metadata = json.loads((benchmark / 'cases.json').read_text())
    selected = [e for e in records['exports'] if e['size'] == records['exports'][0]['size']]
    quality_root = root / 'native-quality'
    quality_root.mkdir()
    for entry in selected + [{'name': 'dis', 'size': None}]:
        name = entry['name']
        if entry['size']:
            w, h = entry['size']
            env['MOTION_ENHANCER_AI_FLOW_MODEL'] = str(benchmark / f'{name}-{w}x{h}.onnx')
        for k, metadata in enumerate(case_metadata):
            folder = benchmark / 'cases' / str(k)
            output = quality_root / f'{name}-{k}.png'
            done = subprocess.run([executable, '--gpu-offline', str(folder / 'a.png'), str(folder / 'b.png'),
                str(output), '--flow-engine', 'dis' if name == 'dis' else 'ai', '--no-ui-mask', '--adapter', 'dgpu'],
                cwd=root, env=env, capture_output=True, timeout=90)
            log = done.stdout.decode('utf-8', errors='replace') + done.stderr.decode('utf-8', errors='replace')
            (quality_root / f'{name}-{k}.log').write_text(log, encoding='utf-8')
            assert done.returncode == 0, log[-2000:]
            if name != 'dis':
                assert 'Flow estimator: AI pyramid flow (ONNX)' in log, 'Silent fallback'
            target = cv2.imread(str(folder / 'truth.png')).astype(np.float64)
            prediction = cv2.imread(str(output)).astype(np.float64)
            a = cv2.imread(str(folder / 'a.png')).astype(np.float64)
            b = cv2.imread(str(folder / 'b.png')).astype(np.float64)
            psnr = lambda x: float(-10 * np.log10(max(float((((x - target) / 255) ** 2).mean()), 1e-10)))
            report['native_quality'].append({'name': name, 'case': k, 'path': metadata['path'],
                'source': metadata['entry']['video'], 'psnr': psnr(prediction), 'blend_psnr': psnr((a + b) * 0.5)})
        (root / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps({'event': 'native_quality_pass', 'name': name, 'cases': len(case_metadata)}), flush=True)
    print(json.dumps({'event': 'done', 'models_checked': len(report['numerical']), 'out': str(root)}), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    main()
