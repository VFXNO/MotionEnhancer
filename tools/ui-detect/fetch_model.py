"""Fetch the optional OmniParser V1 UI detector; never replace an existing model.

The detector is AGPL-3.0, NOT covered by Motion Enhancer's MIT licence.
Requires only Python's standard library. No training or flow-model changes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION = '799bd041b5d053ed44651c2237ced04d8fdb2777'
URL = f'https://huggingface.co/onnx-community/OmniParser-icon_detect_640x640/resolve/{REVISION}/onnx/model_fp16.onnx'
SHA256 = 'cbc393231545eee6155291c7e2527db3eca86ed54935f4155556113344e24d00'
LICENCE_URL = 'https://huggingface.co/microsoft/OmniParser/raw/main/icon_detect/LICENSE'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New ui_detect.onnx path (or a temporary test path).')
    parser.add_argument('--accept-agpl', action='store_true', help='Acknowledge the detector weights use AGPL-3.0.')
    args = parser.parse_args()
    if not args.accept_agpl:
        parser.error('Read the detector licence first; --accept-agpl is required: ' + LICENCE_URL)
    out = args.out.resolve()
    if out.exists():
        if hashlib.sha256(out.read_bytes()).hexdigest() != SHA256:
            parser.error(f'Refusing to overwrite an existing, different file: {out}')
        print(f'Already verified: {out}')
        return
    for suffix in ('.LICENSE.txt', '.source.json'):
        if out.with_suffix(suffix).exists():
            parser.error(f'Refusing to overwrite existing provenance: {out.with_suffix(suffix)}')
    with urllib.request.urlopen(URL, timeout=120) as response:
        model = response.read()
    if hashlib.sha256(model).hexdigest() != SHA256:
        raise RuntimeError('Downloaded model SHA256 did not match the pinned release')
    with urllib.request.urlopen(LICENCE_URL, timeout=60) as response:
        licence = response.read()
    if b'GNU AFFERO GENERAL PUBLIC LICENSE' not in licence or b'Version 3' not in licence:
        raise RuntimeError('Unexpected upstream detector licence')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also protects against another process creating the file.
    with out.open('xb') as file:
        file.write(model)
    with out.with_suffix('.LICENSE.txt').open('xb') as file:
        file.write(licence)
    with out.with_suffix('.source.json').open('x', encoding='utf-8') as file:
        json.dump({'url': URL, 'revision': REVISION, 'sha256': SHA256,
                   'licence': 'AGPL-3.0', 'licence_url': LICENCE_URL,
                   'purpose': 'UI region detection only; NOT a segmentation mask or motion model'}, file, indent=2)
    print(f'Verified {len(model):,} bytes: {out}\nSHA256: {SHA256}')


if __name__ == '__main__':
    main()
