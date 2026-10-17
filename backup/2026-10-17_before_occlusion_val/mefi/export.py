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

from .model import ARCH, Exported, levels_for, load_checkpoint


def emit(**event):
    print(json.dumps(event), flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('checkpoint')
    p.add_argument('--out', required=True)
    p.add_argument('--width', type=int, default=640)
    p.add_argument('--height', type=int, default=352)
    args = p.parse_args(argv)
    # The program copies whole rows of the network input and output, which
    # needs a width of 64 floats (256 bytes) at a time.
    if args.width % 64 or args.height % 16:
        emit(event='error', message='the width must be a multiple of 64 and the height of 16')
        return 1
    try:
        ck, net = load_checkpoint(args.checkpoint)
    except ValueError as e:
        emit(event='error', message=str(e))
        return 1
    model = Exported(net.eval()).eval()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = torch.rand(1, 6, args.height, args.width)
    # One file: the weights inside it, not in a .onnx.data beside it (a copy
    # of the .onnx alone then could not load).
    torch.onnx.export(model, (frames,), args.out, opset_version=18, input_names=['frames'],
                      output_names=['flow', 'mask'], dynamo=True, optimize=True, external_data=False)
    stale = args.out + '.data'
    if os.path.exists(stale):
        os.remove(stale)
    import onnx
    onnx.checker.check_model(onnx.load(args.out))
    # The model learned motion at the scale of its dataset frames: far from
    # it, textures and speeds look unlike anything it saw.
    trained = ck.get('train_height')
    if trained and not 0.67 <= args.height / trained <= 1.5:
        emit(event='warning', message=f'the network height {args.height} is far from the {trained} px frames '
                                      f'it was trained on; build the dataset at about {args.height} px '
                                      f'(or export near {trained} px) for the best results')
    meta = {'format': 'mefi-pyramid-flow-1', 'arch': ARCH, 'width': args.width, 'height': args.height,
            'levels': levels_for(args.height, args.width), 'preset': ck['preset'], 'psnr': ck.get('psnr'),
            'epoch': ck.get('epoch'), 'train_height': trained, 'checkpoint': os.path.abspath(args.checkpoint)}
    with open(os.path.splitext(args.out)[0] + '.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=1)
    emit(event='done', out=os.path.abspath(args.out), **meta)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
