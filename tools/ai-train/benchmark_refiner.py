"""Reproducible, isolated refiner tests using existing training losses only.

Example (from the repository root; output folder MUST NOT already exist):
  tools/ai-train/.venv/Scripts/python.exe tools/ai-train/benchmark_refiner.py \
    --out <new-folder> --executable build-project/Release/motion_enhancer.exe \
    --checkpoint <existing-best.pt> --data <existing-dataset>

No teacher, pseudo-label generation, installation, or user-run continuation.
Short synthetic learning controls do NOT establish real-game quality gains.
Native timing inputs enlarged to 1080p/1440p are timing proxies, not new truth.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

from mefi import export, train
from mefi.data import FullFrames, load_index, split
from mefi.model import ARCH, REFINED_ARCH, PyramidFlow, initialize_from, load_checkpoint


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def statistics(values):
    a = np.asarray(values, dtype=np.float64)
    return {'count': len(a), 'mean_ms': float(a.mean()), 'median_ms': float(np.median(a)),
            'p95_ms': float(np.percentile(a, 95)), 'p99_ms': float(np.percentile(a, 99)),
            'min_ms': float(a.min()), 'max_ms': float(a.max())}


def fixture(seed, batch, h, w):
    """Textured translations, moving foreground and flashes; no flow labels."""
    rng = np.random.default_rng(seed)
    frames = []
    for k in range(batch):
        image = cv2.GaussianBlur(rng.integers(0, 256, (h, w, 3), np.uint8), (5, 5), 0)
        dx = int(rng.choice((-8, -4, -2, 2, 4, 8)))
        dy = int(rng.choice((-2, 0, 2)))
        triplet = []
        for t in range(3):
            transform = np.array([[1, 0, dx * t], [0, 1, dy * t]], np.float32)
            frame = cv2.warpAffine(image, transform, (w, h), borderMode=cv2.BORDER_REFLECT_101)
            # One independently moving solid object with disocclusion.
            x = w // 3 + t * 3
            frame[h // 3:h // 3 + h // 5, x:x + w // 8] = (40, 180, 240)
            triplet.append(frame)
        # Retain return flashes and their original targets in the control.
        if k == 0 and seed % 5 == 0:
            triplet = [np.full_like(image, v) for v in (8, 250, 8)]
        frames.append(np.concatenate(triplet, 2).transpose(2, 0, 1).copy())
    return torch.from_numpy(np.stack(frames)).float().div(255)


@torch.inference_mode()
def score(net, batch):
    with torch.autocast('cuda', dtype=torch.float16):
        flow, mask = net(batch[:, :3], batch[:, 6:9])
    assert torch.isfinite(flow).all() and torch.isfinite(mask).all()
    image = train.middle(batch[:, :3], batch[:, 6:9], flow.float(), mask.float()).clamp(0, 1)
    blend = (batch[:, :3] + batch[:, 6:9]) * 0.5
    target = batch[:, 3:6]
    psnr = lambda x: float((-10 * torch.log10((x - target).square().mean((1, 2, 3)).clamp_min(1e-10))).mean())
    return {'model_psnr': psnr(image), 'blend_psnr': psnr(blend)}


def learn(net, args, validation):
    net = net.cuda().train()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: train.lr_factor(
        s, {'kind': 'cosine', 'steps': args.steps, 'warmup': min(500, args.steps // 20)}))
    scaler = torch.amp.GradScaler('cuda')
    before = score(net, validation)
    timings, losses = [], []
    for step in range(args.steps):
        batch = fixture(100 + step, args.batch, 128, 224).cuda()
        torch.cuda.synchronize()
        start = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.float16):
            loss, _, _ = train.losses(net, batch, occlusion=0)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        assert torch.isfinite(loss) and torch.isfinite(norm), 'Nonfinite training control'
        scaler.step(opt)
        scaler.update()
        sched.step()
        torch.cuda.synchronize()
        timings.append((time.perf_counter() - start) * 1000)
        losses.append(float(loss.detach()))
    after = score(net.eval(), validation)
    learned = float(net.fine[-1].refiner.head.weight.detach().abs().sum()) if net.arch == REFINED_ARCH else None
    if learned is not None:
        assert learned > 0, 'Integrated refiner did not learn'
    return net.cpu(), {'before': before, 'after': after, 'steps': args.steps, 'batch': args.batch,
                       'crop': [224, 128], 'step_timing': statistics(timings[5:] or timings),
                       'step_samples_ms': timings, 'loss': losses, 'refiner_head_abs_sum': learned,
                       'scaler': scaler.state_dict()}


def command(args, root, name, command_args, env):
    done = subprocess.run(command_args, cwd=root, env=env, capture_output=True, timeout=180)
    log = done.stdout.decode('utf-8', errors='replace') + done.stderr.decode('utf-8', errors='replace')
    (root / f'{name}.log').write_text(log, encoding='utf-8')
    if done.returncode:
        raise RuntimeError(f'{name} failed ({done.returncode}): {log[-3000:]}')
    metrics = [json.loads(line[len('BENCH_JSON '):]) for line in log.splitlines() if line.startswith('BENCH_JSON ')]
    return log, metrics


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', required=True)
    p.add_argument('--executable', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--steps', type=int, default=60)
    p.add_argument('--batch', type=int, default=4)
    p.add_argument('--repeats', type=int, default=100)
    p.add_argument('--presets', nargs='+', choices=['small', 'medium', 'large'], default=['small', 'medium'])
    p.add_argument('--sizes', nargs='+', default=['640x352', '512x288'])
    args = p.parse_args(argv)
    if args.steps < 2 or args.batch < 1 or not 1 <= args.repeats <= 200:
        p.error('steps >= 2, batch >= 1 and repeats in 1..200 required')
    sizes = [tuple(map(int, size.split('x'))) for size in args.sizes]
    if any(w < 64 or h < 16 or w % 64 or h % 16 for w, h in sizes):
        p.error('network widths must be positive multiples of 64, heights of 16')
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    args.executable = str(Path(args.executable).resolve())
    args.checkpoint = str(Path(args.checkpoint).resolve())
    args.data = str(Path(args.data).resolve())
    protected = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                 (Path(args.checkpoint), Path(args.data) / 'index.json')}
    write(root / 'protected-hashes.json', protected)
    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    assert torch.cuda.is_available(), 'This benchmark requires the existing GPU PyTorch environment'
    torch.manual_seed(17)
    import onnx
    environment = {'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
                   'torch': str(torch.__version__), 'hip': torch.version.hip, 'cuda': torch.version.cuda,
                   'gpu': torch.cuda.get_device_name(0), 'onnx': onnx.__version__, 'opencv': cv2.__version__,
                   'native_sha256': hashlib.sha256(Path(args.executable).read_bytes()).hexdigest(),
                   'arguments': vars(args), 'teacher': False, 'losses': 'unchanged mefi.train.losses',
                   'gaming_load_tested': False}
    write(root / 'environment.json', environment)
    report = {'environment': environment, 'learning_controls': {}, 'exports': [], 'native': [], 'checks': {}}
    validation = fixture(901, 8, 128, 224).cuda()
    candidates = []
    for preset in args.presets:
        torch.manual_seed(17)
        base = PyramidFlow(preset)
        refined = PyramidFlow(preset, refiner=True)
        refined.load_state_dict(base.state_dict(), strict=False)
        for model in (base, refined):
            name = f'{preset}-{model.arch}'
            trained, metrics = learn(model, args, validation)
            report['learning_controls'][name] = metrics
            candidates.append((name, trained))
            write(root / 'report.json', report)
            print(json.dumps({'event': 'learned', 'name': name, 'before': metrics['before'],
                              'after': metrics['after'], 'step_timing': metrics['step_timing']}), flush=True)
    # Real trained legacy weights also exercise exact, zero-correction migration.
    ck, legacy = load_checkpoint(args.checkpoint)
    refined = PyramidFlow(legacy.preset, refiner=True)
    initialize_from(refined, args.checkpoint)
    with torch.inference_mode():
        test = fixture(905, 1, 128, 224)
        old, new = legacy(test[:, :3], test[:, 6:9]), refined(test[:, :3], test[:, 6:9])
        report['checks']['legacy_initialization_exact'] = all(torch.equal(a, b) for a, b in zip(old, new))
    assert report['checks']['legacy_initialization_exact']
    candidates += [('legacy-base', legacy), ('legacy-refined-zero', refined)]
    items, missing = load_index([args.data])
    _, held = split(items)
    cases = FullFrames(held, limit=12)
    # Real held-out images, kept separate from the synthetic training controls.
    case_root = root / 'cases'
    case_root.mkdir()
    for k, item in enumerate(cases.items):
        folder = case_root / str(k)
        folder.mkdir()
        for i, filename in enumerate(('a.png', 'truth.png', 'b.png')):
            image = cv2.imread(str(Path(item[0]) / f'im{i}.jpg'))
            h, w = image.shape[:2]
            assert cv2.imwrite(str(folder / filename), image[:h // 16 * 16, :w // 16 * 16])
    write(root / 'cases.json', [{'path': i[0], 'entry': i[2]} for i in cases.items])
    for w, h in ((1920, 1080), (2560, 1440)):
        folder = root / f'timing-{w}x{h}'
        folder.mkdir()
        for filename in ('a.png', 'b.png'):
            image = cv2.imread(str(case_root / '0' / filename))
            cv2.imwrite(str(folder / filename), cv2.resize(image, (w, h)))
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    env['MOTION_ENHANCER_BENCH_ITERS'] = str(args.repeats)
    env['MOTION_ENHANCER_BENCH_SERIAL'] = '1'
    env.pop('MOTION_ENHANCER_AI_FLOW_MODEL', None)
    for name, net in candidates:
        checkpoint = root / f'{name}.pt'
        torch.save({'arch': net.arch, 'preset': net.preset, 'model': net.state_dict(),
                    'train_height': ck.get('train_height') if name.startswith('legacy') else 128,
                    'epoch': 0, 'psnr': None}, checkpoint)
        for w, h in sizes:
            # Legacy weights are tested at the proposed 640x352 deployment size.
            if name.startswith('legacy') and (w, h) != sizes[0]:
                continue
            model = root / f'{name}-{w}x{h}.onnx'
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                assert export.main([str(checkpoint), '--out', str(model), '--width', str(w), '--height', str(h)]) == 0
            (root / f'{model.stem}-export.log').write_text(output.getvalue(), encoding='utf-8')
            graph = onnx.load(model)
            assert any(i.data_type == onnx.TensorProto.FLOAT16 for i in graph.graph.initializer)
            report['exports'].append({'name': name, 'size': [w, h], 'parameters': sum(p.numel() for p in net.parameters()),
                                      'onnx_bytes': model.stat().st_size, 'arch': net.arch})
            env['MOTION_ENHANCER_AI_FLOW_MODEL'] = str(model)
            _, network = command(args, root, f'{model.stem}-network',
                                 [args.executable, '--ai-flow-test', str(model), '--adapter', 'dgpu'], env)
            assert network and network[0]['count'] == 200
            record = {'name': name, 'network_size': [w, h], 'network': network, 'pipeline': []}
            for ow, oh in ((1920, 1080), (2560, 1440)):
                folder = root / f'timing-{ow}x{oh}'
                log, metrics = command(args, root, f'{model.stem}-{ow}x{oh}',
                    [args.executable, '--gpu-offline', str(folder / 'a.png'), str(folder / 'b.png'),
                     str(folder / f'{model.stem}.png'), '--flow-engine', 'ai', '--no-ui-mask', '--adapter', 'dgpu'], env)
                assert 'Flow estimator: AI pyramid flow (ONNX)' in log, 'Silent fallback'
                assert len(metrics) == 4 and all(m['count'] == args.repeats for m in metrics)
                record['pipeline'].append({'output_size': [ow, oh], 'statistics': metrics})
            report['native'].append(record)
            write(root / 'report.json', report)
            print(json.dumps({'event': 'benchmarked', 'name': name, 'size': [w, h],
                              'network_mean_ms': network[0]['mean_ms'], 'pipeline': [
                                  {'size': row['output_size'], 'gpu': next(m for m in row['statistics']
                                    if m['stage'] == 'pipeline_gpu_sum')['mean_ms']} for row in record['pipeline']]}), flush=True)
    # No network: DIS comparison through the exact same timing path.
    for w, h in ((1920, 1080), (2560, 1440)):
        folder = root / f'timing-{w}x{h}'
        _, metrics = command(args, root, f'dis-{w}x{h}', [args.executable, '--gpu-offline', str(folder / 'a.png'),
            str(folder / 'b.png'), str(folder / 'dis.png'), '--flow-engine', 'dis', '--no-ui-mask', '--adapter', 'dgpu'], env)
        report['native'].append({'name': 'dis', 'output_size': [w, h], 'statistics': metrics})
    report['checks']['protected_files_unchanged'] = all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        for path, expected in protected.items())
    assert report['checks']['protected_files_unchanged']
    write(root / 'report.json', report)
    print(json.dumps({'event': 'done', 'out': str(root), 'checks': report['checks']}), flush=True)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    raise SystemExit(main())
