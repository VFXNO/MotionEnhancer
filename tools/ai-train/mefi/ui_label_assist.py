"""Optional local binary-segmentation guidance, not accepted training truth.

Only an explicitly selected, checksum-pinned ONNX model is loaded. The CPU
OpenCV backend requires no additional runtime package or network connection.
"""
from pathlib import Path

import cv2
import numpy as np

from .ui_labeling import digest_bytes, file_hash


NORMALIZATIONS = ('RGB 0..1', 'RGB -1..1', 'RGB ImageNet')


def inspect_model(path, expected):
    import onnx
    path = Path(path).resolve()
    if path.suffix.lower() != '.onnx' or not path.is_file() or path.stat().st_size > 512 * 1024 * 1024:
        raise ValueError('Choose a self-contained ONNX model, at most 512 MiB')
    raw = path.read_bytes()
    if digest_bytes(raw) != expected:
        raise ValueError('The AI model checksum changed; select it again')
    model = onnx.load_model_from_string(raw)
    def graph_check(graph):
        tensors = list(graph.initializer)
        for sparse in graph.sparse_initializer:
            tensors += [sparse.values, sparse.indices]
        for node in graph.node:
            if node.domain not in ('', 'ai.onnx'):
                raise ValueError('Custom operator domains are not supported for label guidance')
            for attr in node.attribute:
                if attr.HasField('t'):
                    tensors.append(attr.t)
                tensors.extend(attr.tensors)
                if attr.HasField('g'):
                    graph_check(attr.g)
                for nested in attr.graphs:
                    graph_check(nested)
        if any(t.external_data or t.data_location == onnx.TensorProto.EXTERNAL for t in tensors):
            raise ValueError('External ONNX tensor files are not supported; use a self-contained model')
    graph_check(model.graph)
    if model.functions:
        raise ValueError('ONNX models containing local functions are not supported by this assistant')
    initializers = {tensor.name for tensor in model.graph.initializer}
    inputs = [item for item in model.graph.input if item.name not in initializers]
    if len(inputs) != 1 or len(model.graph.output) != 1:
        raise ValueError('Use a binary dense segmenter with one RGB input and one UI output; box/OCR/flow/prompt models need an adapter')
    def shape(item):
        tensor = item.type.tensor_type
        if tensor.elem_type != onnx.TensorProto.FLOAT:
            raise ValueError('The assistant requires FP32 input/output tensors')
        return [d.dim_value if d.HasField('dim_value') else None for d in tensor.shape.dim]
    inp, output = shape(inputs[0]), shape(model.graph.output[0])
    if len(inp) != 4 or inp[:2] != [1, 3] or any(not isinstance(d, int) or not 16 <= d <= 4096 for d in inp[2:]):
        raise ValueError('Expected fixed RGB NCHW input [1,3,H,W], with H/W between 16 and 4096')
    if len(output) != 4 or output[:2] != [1, 1] or any(not isinstance(d, int) or not 1 <= d <= 4096 for d in output[2:]):
        raise ValueError('Expected one dense UI output [1,1,H,W], not boxes or multiclass predictions')
    onnx.checker.check_model(model)
    return {'path': str(path), 'sha256': expected, 'input': inputs[0].name,
            'output': model.graph.output[0].name, 'input_shape': inp, 'output_shape': output}


def probability_map(raw, output_kind, original_shape):
    raw = np.asarray(raw)
    if raw.ndim != 4 or raw.shape[:2] != (1, 1) or not np.isfinite(raw).all():
        raise ValueError('The model did not produce a finite dense [1,1,H,W] UI map')
    value = raw[0, 0].astype(np.float32)
    if output_kind == 'logits':
        value = 1 / (1 + np.exp(-np.clip(value, -80, 80)))
    elif output_kind != 'probabilities' or value.min() < 0 or value.max() > 1:
        raise ValueError('Select the correct model output type: probabilities [0,1] or logits')
    h, w = original_shape
    return np.clip(cv2.resize(value, (w, h), interpolation=cv2.INTER_LINEAR), 0, 1)


def suggest(frame, model_path, expected, *, normalization='RGB 0..1', output_kind='probabilities'):
    if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3 or normalization not in NORMALIZATIONS:
        raise ValueError('Invalid RGB frame or model normalization')
    info = inspect_model(model_path, expected)
    h, w = info['input_shape'][2:]
    # Explicitly stretch only the inference copy; map output back to the exact
    # prepared annotation coordinates. The project frames are not changed.
    image = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255
    if normalization == 'RGB -1..1':
        image = image * 2 - 1
    elif normalization == 'RGB ImageNet':
        image = (image - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    blob = np.ascontiguousarray(image.transpose(2, 0, 1)[None])
    raw = Path(model_path).read_bytes()
    if digest_bytes(raw) != expected:
        raise ValueError('AI model changed before inference; result discarded')
    net = cv2.dnn.readNetFromONNX(np.frombuffer(raw, np.uint8))
    cv2.ocl.setUseOpenCL(False)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    net.setInput(blob, info['input'])
    output = net.forward(info['output'])
    if list(output.shape) != info['output_shape']:
        raise ValueError('AI output dimensions differ from its declared model contract')
    result = probability_map(output, output_kind, frame.shape[:2])
    if file_hash(model_path) != expected:
        raise ValueError('AI model changed during inference; result discarded')
    info.update(normalization=normalization, output_kind=output_kind,
                backend='OpenCV DNN CPU', resize='stretch inference copy, bilinear output restore',
                role='unreviewed suggestion, not ground truth')
    return result, info
