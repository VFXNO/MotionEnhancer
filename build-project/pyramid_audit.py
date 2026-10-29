"""Exercise the existing executable without modifying application sources."""
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'build-project' / 'pyramid-audit'
OUT.mkdir(exist_ok=True)
EXE = ROOT / 'build-project' / 'Release' / 'motion_enhancer.exe'
env = os.environ.copy()
env['MOTION_ENHANCER_NEURAL_MODEL'] = str(ROOT / 'neural_model.bin')
for key in ('MOTION_ENHANCER_BENCH_ITERS', 'MOTION_ENHANCER_D3D_DEBUG',
            'MOTION_ENHANCER_SAD_DEBUG'):
    env.pop(key, None)
results = []

def run(name, args, extra_env=None):
    current_env = env.copy()
    if extra_env:
        current_env.update(extra_env)
    try:
        p = subprocess.run([str(EXE), *map(str, args)], cwd=ROOT, env=current_env,
                           capture_output=True, text=True, errors='replace', timeout=120)
        log = p.stdout + p.stderr
        code = p.returncode
    except subprocess.TimeoutExpired as e:
        log = f'TIMEOUT: {e}'
        code = 'timeout'
    (OUT / (name + '.log')).write_text(log, encoding='utf-8')
    levels = re.search(r'(\d+) pyramid levels', log)
    score = re.search(r'PSNR vs ground truth:\s+([\d.]+)', log)
    engine = re.search(r'Flow estimator: (.+)', log)
    result = dict(name=name, exit=code, effective_levels=int(levels[1]) if levels else None,
                  psnr=float(score[1]) if score else None, engine=engine[1] if engine else None)
    results.append(result)
    (OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)
    return result

def save(array, path):
    Image.fromarray(array).save(path, compress_level=1)

def shifted(array, dx, dy):
    h, w = array.shape[:2]
    return array[np.clip(np.arange(h) - dy, 0, h - 1)[:, None],
                 np.clip(np.arange(w) - dx, 0, w - 1)[None, :]]

source = Image.open(ROOT / 'flow_tests' / 'hd0.png').convert('RGB')
for w, h in ((320, 240), (1280, 720), (1920, 1080), (2560, 1440),
             (3840, 2160), (4097, 2049), (1280, 1023), (1280, 1024),
             (3072, 2047), (3072, 2048)):
    tag = f'{w}x{h}'
    base = np.array(source.resize((w, h), Image.Resampling.BICUBIC))
    a, b, gt = (OUT / f'{tag}_{s}.png' for s in ('a', 'b', 'gt'))
    save(base, a)
    save(shifted(base, 48, 16), b)
    save(shifted(base, 24, 8), gt)
    engines = ('dis', 'sad', 'ffx', 'neural') if (w, h) in ((1920, 1080), (3840, 2160)) else ('dis',)
    for engine in engines:
        for levels in ((8, 10) if len(engines) > 1 else (10,)):
            name = f'{tag}_{engine}_{levels}'
            output = OUT / (name + '.png')
            result = run(name, ['--gpu-offline', a, b, output, '--flow-engine', engine,
                                '--gpu-levels', levels, '--ground-truth', gt])
            if result['exit'] == 0:
                with Image.open(output) as im:
                    result['pixel_hash'] = hashlib.sha256(im.tobytes()).hexdigest()
    if (w, h) == (3840, 2160):
        for engine in ('dis', 'sad', 'ffx', 'neural'):
            name = f'{tag}_static_{engine}_10'
            output = OUT / (name + '.png')
            result = run(name, ['--gpu-offline', a, a, output, '--flow-engine', engine,
                                '--gpu-levels', 10, '--ground-truth', a])
            if result['exit'] == 0:
                actual = np.array(Image.open(output)).astype(np.int16)
                difference = np.abs(actual - base.astype(np.int16))
                result['max_static_pixel_error'] = int(difference.max())
                result['mean_static_pixel_error'] = float(difference.mean())
        # Exposure change plus motion: test brightness-statistics sensitivity.
        save(np.clip(shifted(base, 48, 16).astype(np.float32) * 1.12 + 10, 0, 255).astype(np.uint8), b)
        save(np.clip(shifted(base, 24, 8).astype(np.float32) * 1.06 + 5, 0, 255).astype(np.uint8), gt)
        for engine in ('dis', 'sad', 'ffx'):
            for levels in (8, 10):
                name = f'{tag}_exposure_{engine}_{levels}'
                run(name, ['--gpu-offline', a, b, OUT / (name + '.png'), '--flow-engine', engine,
                           '--gpu-levels', levels, '--ground-truth', gt])
    del base
    gc.collect()

# Actual temporal chain: motion starts, stops, reverses and stops again.
w, h = 3072, 2048
base = np.array(source.resize((w, h), Image.Resampling.BICUBIC))
prefix = OUT / 'temporal'
offsets = (0, 0, 48, 96, 96, 48, 0, 0)
for i, x in enumerate(offsets):
    save(shifted(base, x, 0), Path(f'{prefix}_f{i:03}.png'))
    if i + 1 < len(offsets):
        save(shifted(base, (x + offsets[i + 1]) // 2, 0), Path(f'{prefix}_m{i:03}.png'))
del base
gc.collect()
for engine in ('dis', 'sad', 'ffx', 'neural'):
    for levels in (8, 10):
        run(f'temporal_{engine}_{levels}', ['--gpu-offline-sequence', prefix,
                                            '--flow-engine', engine, '--gpu-levels', levels])

# The parity tool has its own fixed settings; a small input exercises the new 4 px floor.
run('neural_parity_small', ['--neural-parity-check', ROOT / 'flow_tests' / 'cel0.png',
                          ROOT / 'flow_tests' / 'cel1.png'])
run('gpu_selftest_debug', ['--d3d12-self-test'], {'MOTION_ENHANCER_D3D_DEBUG': '2'})
(OUT / 'results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
print(f'COMPLETE: {len(results)} executions; results in {OUT}', flush=True)
