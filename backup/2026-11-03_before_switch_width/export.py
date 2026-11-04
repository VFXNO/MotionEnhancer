"""Exports a trained checkpoint to the ONNX model Motion Enhancer runs.

    python -m mefi.export runs/mygame/best.pt --out ../../build/Release/models/ai_flow.onnx

The network runs at a fixed resolution (--width x --height, multiples of
16): the program scales each captured frame to it. A quarter of 1440p
(640x352) is a starting point; benchmark network and full-pipeline latency.
Input  "frames" [1, 6, H, W] float: frame0 RGB then frame1 RGB, 0..1.
Output "flow"   [1, 4, H, W] float: F01 (x, y), F10 (x, y), network pixels.
Output "mask"   [1, 1, H, W] float: the middle frame's share of frame0.
A <name>.json next to it records the resolution, preset and score.
Refined checkpoints also export <name>.dis.onnx/.json with the integrated
head removed, allowing the programme to select DIS refinement instead.
"""
import argparse
import json
import os
import uuid

import torch

from .model import Exported, REFINED_ARCH, levels_for, load_checkpoint
from .checkpoint import save_config


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
    if args.width < 64 or args.height < 16 or args.width % 64 or args.height % 16:
        emit(event='error', message='the width must be a multiple of 64 and the height of 16')
        return 1
    try:
        checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        if isinstance(checkpoint, dict) and checkpoint.get('format') == 'mefi-ui-training-1':
            from .export_ui import export_checkpoint
            return export_checkpoint(checkpoint, args, emit)
        meta_path = os.path.splitext(args.out)[0] + '.json'
        if os.path.isfile(meta_path):
            with open(meta_path, encoding='utf-8') as handle:
                if json.load(handle).get('format') == 'mefi-pyramid-ui-1':
                    raise ValueError('This output belongs to a temporal UI bundle. Choose a separate flow-only filename.')
        ck, net = load_checkpoint(args.checkpoint)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as e:
        emit(event='error', message=str(e))
        return 1
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = torch.rand(1, 6, args.height, args.width)
    refined = net.arch == REFINED_ARCH
    bundle_id = uuid.uuid4().hex
    stem = os.path.splitext(args.out)[0]
    dis_out = stem + '.dis.onnx'
    # Make the bypass before Exported converts the original weights to FP16.
    variants = [(dis_out, net.without_refiner(), False), (args.out, net, True)] if refined else [(args.out, net, False)]
    for out, variant, integrated in variants:
        model = Exported(variant.eval()).eval()
        # Embedded weights: no .onnx.data dependency. The bypass graph has
        # no refiner operators, rather than computing and discarding them.
        torch.onnx.export(model, (frames,), out, opset_version=18, input_names=['frames'],
                          output_names=['flow', 'mask'], dynamo=True, optimize=True, external_data=False)
        stale = out + '.data'
        if os.path.exists(stale):
            os.remove(stale)
        import onnx
        onnx.checker.check_model(onnx.load(out))
        meta = {'format': 'mefi-pyramid-flow-1', 'arch': variant.arch, 'width': args.width, 'height': args.height,
                'levels': levels_for(args.height, args.width), 'preset': ck['preset'], 'psnr': ck.get('psnr'),
                'epoch': ck.get('epoch'), 'train_height': ck.get('train_height'), 'validation': ck.get('validation'),
                'checkpoint': os.path.abspath(args.checkpoint), 'trained_arch': net.arch,
                'integrated_refiner': integrated, 'bundle_id': bundle_id}
        if refined:
            meta['dis_model'] = os.path.basename(dis_out)
            meta['refined_model'] = os.path.basename(args.out)
            if not integrated:
                meta['psnr'] = None  # Saved validation score does not score this bypass.
        # Publish the primary metadata last. Mixed/partial pairs fail the
        # bundle-ID check rather than silently selecting stale weights.
        save_config(meta, os.path.splitext(out)[0] + '.json')
        emit(event='variant', out=os.path.abspath(out), integrated_refiner=integrated)
    # The model learned motion at the scale of its dataset frames: far from
    # it, textures and speeds look unlike anything it saw.
    trained = ck.get('train_height')
    if trained and not 0.67 <= args.height / trained <= 1.5:
        emit(event='warning', message=f'the network height {args.height} is far from the {trained} px frames '
                                      f'it was trained on; build the dataset at about {args.height} px '
                                      f'(or export near {trained} px) for the best results')
    emit(event='done', out=os.path.abspath(args.out), dis_out=os.path.abspath(dis_out) if refined else None,
         runtime_name='ai_flow.onnx', **meta)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
