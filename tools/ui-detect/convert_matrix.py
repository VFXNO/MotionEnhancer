"""Convert the pinned UI detector to a SEPARATE Vulkan matrix backend graph.

Requires onnx and the pnnx 20261020 executable. No training/calibration, no flow
models, no runtime replacement. Output must be a new directory. The detector
weights retain AGPL-3.0; ncnn/pnnx are separately BSD-3-Clause.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import onnx

SOURCE_SHA256 = '0432face5e01e085eb3fc1b2b64ca706cc7916cd18a48baa5fdc55a4bfe88ef4'
PNNX_SHA256 = 'b8c56a0b0319e6ce4d229e9a339e4ee377fb0c42f7f3e66eec235ed00cd71b18'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Pinned upstream FP32 conversion, not a flow model.')
    parser.add_argument('--pnnx', type=Path, required=True)
    parser.add_argument('--licence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error('Wrong source hash; refusing to convert another model.')
    if args.out.exists():
        raise FileExistsError('Output exists; refusing to replace model/provenance files')
    licence = args.licence.read_bytes()
    if b'GNU AFFERO GENERAL PUBLIC LICENSE' not in licence:
        parser.error('Expected the upstream detector AGPL licence.')
    if hashlib.sha256(args.pnnx.read_bytes()).hexdigest() != PNNX_SHA256:
        parser.error('Expected the pinned pnnx 20261020 Windows x64 executable.')
    model = onnx.load(args.model)
    onnx.checker.check_model(model)
    args.out.mkdir(parents=True, exist_ok=False)
    source = args.out.resolve() / 'source.onnx'
    shutil.copy2(args.model, source)
    result = subprocess.run([str(args.pnnx.resolve()), str(source), 'inputshape=[1,3,640,640]', 'fp16=1'],
                            cwd=args.out.resolve(), capture_output=True, timeout=120)
    log = (result.stdout + result.stderr).decode('utf-8', errors='replace')
    (args.out / 'conversion.log').write_text(log, encoding='utf-8')
    if result.returncode:
        raise RuntimeError('pnnx conversion failed; see conversion.log')
    param, weights = args.out / 'ui_detect.matrix.param', args.out / 'ui_detect.matrix.bin'
    shutil.copy2(args.out / 'source.ncnn.param', param)
    shutil.copy2(args.out / 'source.ncnn.bin', weights)
    (args.out / 'ui_detect.matrix.LICENSE.txt').write_bytes(licence)
    (args.out / 'ui_detect.matrix.source.json').write_text(json.dumps({
        'source_sha256': SOURCE_SHA256,
        'source_url': 'https://huggingface.co/onnx-community/OmniParser-icon_detect_640x640/resolve/799bd041b5d053ed44651c2237ced04d8fdb2777/onnx/model.onnx',
        'pnnx_version': '20261020', 'pnnx_exe_sha256': PNNX_SHA256,
        'param_sha256': hashlib.sha256(param.read_bytes()).hexdigest(),
        'bin_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'licence': 'AGPL-3.0', 'purpose': 'UI regions only, not pixel masking or motion',
        'precision': 'FP16 weights/storage, FP32 matrix accumulation',
        'training': False, 'runtime_verification_required': True
    }, indent=2), encoding='utf-8')
    print(f'Separate matrix graph created: {param}. Not activated; native ISA/accuracy/latency tests required.')


if __name__ == '__main__':
    main()
