"""Additional tests; run after pyramid_audit.py has finished."""
import json
import os
from pathlib import Path
import subprocess
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'build-project' / 'pyramid-audit'
EXE = ROOT / 'build-project' / 'Release' / 'motion_enhancer.exe'
env = os.environ.copy()
env['MOTION_ENHANCER_NEURAL_MODEL'] = str(ROOT / 'neural_model.bin')
env.pop('MOTION_ENHANCER_D3D_DEBUG', None)
env.pop('MOTION_ENHANCER_BENCH_ITERS', None)
results = []

def run(name, args, debug=None):
    e = env.copy()
    if debug:
        e['MOTION_ENHANCER_D3D_DEBUG'] = debug
    try:
        p = subprocess.run([str(EXE), *map(str, args)], cwd=ROOT, env=e,
                           capture_output=True, text=True, errors='replace', timeout=120)
        text = p.stdout + p.stderr
        code = p.returncode
    except subprocess.TimeoutExpired as exc:
        text, code = str(exc), 'timeout'
    (OUT / (name + '.log')).write_text(text, encoding='utf-8')
    results.append({'name': name, 'exit': code})
    print(name, code, text, flush=True)
    (OUT / 'extra-results.json').write_text(json.dumps(results, indent=2), encoding='utf-8')

# Existing native-resolution scenes: pans, rapid pans, text, moving objects.
for scene in ('big', 'big24_', 'big48_', 'big96_', 'big160_', 'hd', 'L', 'pan', 'txt', 'txtov'):
    a, b = (ROOT / 'flow_tests' / f'{scene}{i}.png' for i in (0, 1))
    gt = ROOT / 'flow_tests' / (scene.rstrip('_') + '_gt.png')
    if not all(p.exists() for p in (a, b, gt)):
        continue
    for engine in ('dis', 'sad'):
        for levels in (8, 10):
            name = f'existing_{scene}_{engine}_{levels}'
            run(name, ['--gpu-offline', a, b, OUT / (name + '.png'), '--ground-truth', gt,
                       '--flow-engine', engine, '--gpu-levels', levels])

# AI model and raw FFX: independently check that they are really active.
for engine, options in (('ai', []), ('ffx', ['--no-ffx-refine']),
                         ('ffx', ['--ffx-occlusion'])):
    for levels in (8, 10):
        name = f'extra_{engine}_{"_".join(options)}_{levels}'
        run(name, ['--gpu-offline', OUT / '3072x2048_a.png', OUT / '3072x2048_b.png',
                   OUT / (name + '.png'), '--ground-truth', OUT / '3072x2048_gt.png',
                   '--flow-engine', engine, '--gpu-levels', levels, *options])

# A static input must not acquire phantom motion merely from deeper recursion.
for engine in ('dis', 'sad', 'ffx', 'neural'):
    name = f'static_baseline_{engine}_8'
    run(name, ['--gpu-offline', OUT / '3840x2160_a.png', OUT / '3840x2160_a.png',
               OUT / (name + '.png'), '--ground-truth', OUT / '3840x2160_a.png',
               '--flow-engine', engine, '--gpu-levels', 8])

# Validation at a true ten-level size, plus odd-sized coarse levels.
for engine in ('dis', 'sad', 'ffx', 'neural'):
    name = f'debug_ten_{engine}'
    run(name, ['--gpu-offline', OUT / '3072x2048_a.png', OUT / '3072x2048_b.png',
               OUT / (name + '.png'), '--flow-engine', engine, '--gpu-levels', 10], debug='1')
for engine in ('sad', 'ffx', 'neural'):
    name = f'odd_ten_{engine}'
    run(name, ['--gpu-offline', OUT / '4097x2049_a.png', OUT / '4097x2049_b.png',
               OUT / (name + '.png'), '--ground-truth', OUT / '4097x2049_gt.png',
               '--flow-engine', engine, '--gpu-levels', 10])

# This accepted option combination has an effective level count smaller than min-refine.
run('min_refine_effective_mismatch', ['--gpu-offline', ROOT / 'flow_tests' / 'hd0.png',
    ROOT / 'flow_tests' / 'hd1.png', OUT / 'min-refine.png', '--ground-truth',
    ROOT / 'flow_tests' / 'hd_gt.png', '--flow-engine', 'sad', '--gpu-levels', 10, '--gpu-min-refine', 9])

print(f'COMPLETE: {len(results)} additional executions', flush=True)
