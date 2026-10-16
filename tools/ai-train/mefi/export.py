"""Exports a trained checkpoint to the ONNX model Motion Enhancer runs.

    python -m mefi.export runs/mygame/best.pt --out ../../build/Release/models/ai_flow.onnx

The network runs at a fixed resolution (--width x --height, multiples of
16): the program scales each captured frame to it. A quarter of 1440p
(640x352) keeps the network around a millisecond on an RX 9070 XT.
Input  "frames" [1, 6, H, W] float: frame0 RGB then frame1 RGB, 0..1.
Output "flow"   [1, 4, H, W] float: F01 (x, y), F10 (x, y), network pixels.
Output "mask"   [1, 1, H, W] float: the middle frame's share of frame0.
A <name>.json next to it records the resolution, preset and score.
"""
import argparse
import json
import os

import torch

from .model import Exported, PyramidFlow


def emit(**event):
    print(json.dumps(event), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=352)
    args = p.parse_args(argv)
    if args.width % 16 or args.height % 16:
        emit(event='error', message='width and height must be multiples of 16')
        return 1
    ck = torch.load(args.checkpoint, map_location='cpu')
    net = PyramidFlow(ck['preset'])
    net.load_state_dict(ck['model'])
    model = Exported(net.eval()).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = torch.rand(1, 6, args.height, args.width)
    torch.onnx.export(model, (frames,), args.out, opset_version=18, input_names=['frames'],
                      output_names=['flow', 'mask'], dynamo=True, optimize=True)
    import onnx
    onnx.checker.check_model(onnx.load(args.out))
    meta = {'format': 'mefi-pyramid-flow-1', 'width': args.width, 'height': args.height,
            'preset': ck['preset'], 'psnr': ck.get('psnr'), 'epoch': ck.get('epoch'),
            'checkpoint': os.path.abspath(args.checkpoint)}
    with open(os.path.splitext(args.out)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    emit(event='done', out=os.path.abspath(args.out), **meta)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
