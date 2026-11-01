"""Fetch the pinned BSD-3-Clause ncnn source for explicit Vulkan matrix inference.

Does not build, train, download weights, activate a model or modify installation.
Existing destinations are never overwritten. Use the canonical build-project.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import urllib.request
import zipfile

VERSION = '20261020'
URL = f'https://github.com/Tencent/ncnn/releases/download/{VERSION}/ncnn-{VERSION}-full-source.zip'
SHA256 = '699d9acc67489eee0993a551ca1ebc247bbf0f4e33babe1dec810e6a9a52b988'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('third_party/ui-matrix/ncnn'))
    args = parser.parse_args()
    if args.out.exists() or (args.out.parent / 'ncnn.source.json').exists():
        parser.error('Destination exists; refusing to overwrite it.')
    data = urllib.request.urlopen(URL, timeout=120).read()
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError('ncnn source hash mismatch')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        root = args.out.resolve()
        for info in archive.infolist():
            if not (root / info.filename).resolve().is_relative_to(root):
                raise RuntimeError('Unsafe archive path')
        args.out.mkdir(parents=True, exist_ok=False)
        archive.extractall(args.out)
    (args.out.parent / 'ncnn.source.json').write_text(json.dumps({
        'version': VERSION, 'url': URL, 'sha256': SHA256, 'licence': 'BSD-3-Clause'
    }, indent=2), encoding='utf-8')
    print(f'Verified ncnn {VERSION} source: {args.out}')


if __name__ == '__main__':
    main()
