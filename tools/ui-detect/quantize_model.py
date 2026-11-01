"""Create a SEPARATE static-INT8 UI detector, without training or model replacement.

Requires onnx, onnxruntime (quantizer/CPU calibration only), numpy, Pillow and
OpenCV. Calibration reads only explicitly supplied screenshots. The output
directory must be new. Native inference/placement must be tested separately.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
import onnx
from PIL import Image
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod,
                                      QuantFormat, QuantType, quantize_static)
from onnxruntime.quantization.shape_inference import quant_pre_process

SOURCE_SHA256 = '0432face5e01e085eb3fc1b2b64ca706cc7916cd18a48baa5fdc55a4bfe88ef4'
SOURCE_URL = ('https://huggingface.co/onnx-community/OmniParser-icon_detect_640x640/'
              'resolve/799bd041b5d053ed44651c2237ced04d8fdb2777/onnx/model.onnx')


def tensor(path):
    # Same RGB, half-pixel bilinear sampling and integer centred letterbox as
    # UiDetectInputCS. No ImageNet normalisation, sharpening or data augmentation.
    image = np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0
    height, width = image.shape[:2]
    scale = min(640 / width, 640 / height)
    rw, rh = max(1, int(width * scale + 0.5)), max(1, int(height * scale + 0.5))
    left, top = (640 - rw) // 2, (640 - rh) // 2
    canvas = np.full((640, 640, 3), 114 / 255.0, dtype=np.float32)
    canvas[top:top + rh, left:left + rw] = cv2.resize(image, (rw, rh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1)[None])


class Screenshots(CalibrationDataReader):
    def __init__(self, paths):
        self.paths = list(paths)
        self.rewind()

    def get_next(self):
        path = next(self.iterator, None)
        return None if path is None else {'images': tensor(path)}

    def rewind(self):
        self.iterator = iter(self.paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True, help='Pinned upstream FP32 ONNX source, never FP16.')
    parser.add_argument('--licence', type=Path, required=True, help='Existing upstream detector AGPL licence.')
    parser.add_argument('--out', type=Path, required=True, help='NEW output directory.')
    parser.add_argument('--format', choices=['qoperator', 'qdq'], default='qoperator')
    parser.add_argument('--activation', choices=['uint8', 'int8'], default='uint8')
    parser.add_argument('images', nargs='+', type=Path)
    args = parser.parse_args()
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != SOURCE_SHA256:
        parser.error('Source does not match the pinned upstream FP32 UI detector; refusing to quantize another model.')
    licence = args.licence.read_bytes()
    if b'GNU AFFERO GENERAL PUBLIC LICENSE' not in licence:
        parser.error('Expected the upstream AGPL detector licence.')
    for path in args.images:
        if not path.is_file():
            parser.error(f'Calibration image missing: {path}')
    args.out.mkdir(parents=True, exist_ok=False)
    prepared = args.out / 'source.prepared.onnx'
    quant_pre_process(str(args.model.resolve()), str(prepared.resolve()), skip_optimization=True)
    target = args.out / 'ui_detect.int8.onnx'
    quantize_static(str(prepared), str(target), Screenshots(args.images),
                    quant_format=QuantFormat.QOperator if args.format == 'qoperator' else QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8 if args.activation == 'uint8' else QuantType.QInt8,
                    weight_type=QuantType.QInt8, per_channel=True,
                    op_types_to_quantize=['Conv'], calibrate_method=CalibrationMethod.MinMax,
                    calibration_providers=['CPUExecutionProvider'],
                    extra_options={'WeightSymmetric': True})
    model = onnx.load(target)
    onnx.checker.check_model(model)
    operators = dict(Counter(n.op_type for n in model.graph.node))
    # No dynamic per-layer range scans, and genuine 8-bit convolution weights.
    if operators.get('DynamicQuantizeLinear') or not any(t.data_type == onnx.TensorProto.INT8 for t in model.graph.initializer):
        raise RuntimeError('Expected static INT8 weights/activations, not a dynamic or float-only export')
    properties = {p.key: p.value for p in model.metadata_props}
    properties.update({'ui_detector.precision': 'static-int8-' + args.format,
                       'ui_detector.licence': 'AGPL-3.0', 'ui_detector.source_sha256': SOURCE_SHA256})
    onnx.helper.set_model_props(model, properties)
    onnx.save(model, target)
    shutil.copy2(args.licence, target.with_suffix('.LICENSE.txt'))
    report = {'source_url': SOURCE_URL, 'source_sha256': SOURCE_SHA256,
              'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'licence': 'AGPL-3.0',
              'precision': 'static-int8', 'format': args.format, 'activations': args.activation,
              'weights': 'int8-per-channel', 'calibration': 'MinMax; no training',
              'onnxruntime_version': ort.__version__, 'operators': operators,
              'calibration_inputs': [{'path': str(p.resolve()), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                     for p in args.images],
              'matrix_instructions_verified': False, 'native_gpu_benchmark_required': True}
    target.with_suffix('.source.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
