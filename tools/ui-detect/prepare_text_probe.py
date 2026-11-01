"""Prepare a pinned local text model for the standalone OFFLINE probe only.

No downloads, activation, training or environment changes. Uses the existing
isolated onnxruntime.transformers converter when it is on PYTHONPATH.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def topological_sort(model):
    ready = {v.name for v in model.graph.input} | {v.name for v in model.graph.initializer}
    pending = list(model.graph.node)
    ordered = []
    while pending:
        waiting = []
        for node in pending:
            if all(not name or name in ready for name in node.input):
                ordered.append(node)
                ready.update(node.output)
            else:
                waiting.append(node)
        if len(waiting) == len(pending):
            raise ValueError('Cannot topologically order converted graph')
        pending = waiting
    del model.graph.node[:]
    model.graph.node.extend(ordered)
    return model


def wrap(model, kind, size, height=None):
    height = height or size
    if len(model.graph.input) != 1 or len(model.graph.output) != (2 if kind == 'craft' else 1):
        raise ValueError('Unexpected source I/O contract')
    if model.functions or any(n.domain not in ('', 'ai.onnx') for n in model.graph.node):
        raise ValueError('Only standard ONNX graphs are allowed')
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
        raise ValueError('External model data is not allowed in this probe')
    old = model.graph.input[0].name
    for node in model.graph.node:
        for i, key in enumerate(node.input):
            if key == old:
                node.input[i] = 'probe_normalized'
    model.graph.input[0].name = 'probe_rgb'
    for dim, value in zip(model.graph.input[0].type.tensor_type.shape.dim, (1, 3, height, size)):
        dim.ClearField('dim_param')
        dim.dim_value = value
    if kind == 'craft':
        del model.graph.output[1:]
        output_shape = (1, height // 2, size // 2, 2)
    else:
        output_shape = (1, 1, height, size)
    for dim, value in zip(model.graph.output[0].type.tensor_type.shape.dim, output_shape):
        dim.ClearField('dim_param')
        dim.dim_value = value
    del model.graph.value_info[:]
    nodes = []
    source = 'probe_rgb'
    if kind == 'db':
        model.graph.initializer.append(numpy_helper.from_array(np.array([2, 1, 0], np.int64), 'probe_bgr_indices'))
        nodes.append(helper.make_node('Gather', [source, 'probe_bgr_indices'], ['probe_bgr'], axis=1))
        source = 'probe_bgr'
    model.graph.initializer.extend([
        numpy_helper.from_array(np.array([.485, .456, .406], np.float32).reshape(1, 3, 1, 1), 'probe_mean'),
        numpy_helper.from_array(np.array([.229, .224, .225], np.float32).reshape(1, 3, 1, 1), 'probe_std')])
    nodes.extend([helper.make_node('Sub', [source, 'probe_mean'], ['probe_centered']),
                  helper.make_node('Div', ['probe_centered', 'probe_std'], ['probe_normalized'])])
    original = list(model.graph.node)
    del model.graph.node[:]
    model.graph.node.extend(nodes + original)
    return onnx.shape_inference.infer_shapes(model)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--kind', choices=('db', 'craft'), required=True)
    parser.add_argument('--size', type=int, default=640)
    parser.add_argument('--height', type=int, help='Input height; defaults to --size (square)')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.size < 320 or args.size > 1280 or args.size % 32:
        parser.error('--size must be a multiple of 32 in 320..1280')
    height = args.height or args.size
    if height < 320 or height > 1280 or height % 32:
        parser.error('--height must be a multiple of 32 in 320..1280')
    if args.out.exists():
        raise FileExistsError('Refusing to replace an existing output folder')
    digest = hashlib.sha256(args.source.read_bytes()).hexdigest()
    if digest != args.sha256:
        raise ValueError('Source SHA256 mismatch')
    from onnxruntime.transformers.float16 import convert_float_to_float16
    model = wrap(onnx.load(args.source, load_external_data=False), args.kind, args.size, height)
    onnx.checker.check_model(model)
    variants = {'fp32': model,
                'fp16': topological_sort(convert_float_to_float16(copy.deepcopy(model), keep_io_types=True,
                                                                 disable_shape_infer=True))}
    for candidate in variants.values():
        onnx.checker.check_model(candidate)
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for precision, candidate in variants.items():
        path = args.out / f'model-{precision}.onnx'
        onnx.save(candidate, path)
        records.append({'precision': precision, 'path': str(path.resolve()),
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    report = {'source': str(args.source.resolve()), 'source_sha256': digest, 'kind': args.kind,
              'input_width': args.size, 'input_height': height, 'variants': records, 'offline_only': True, 'activated': False,
              'changes': 'Static shape, GPU source-specific normalization, optional FP16; no training/calibration'}
    (args.out / 'preparation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
