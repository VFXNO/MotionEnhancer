"""INT8 comparison/calibration safety tests; all generated files are temporary."""
import contextlib
import hashlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import benchmark_int8


class ComparisonTests(unittest.TestCase):
    def test_percentiles_and_deadline_counts(self):
        values = [9, 1, 4, 2]
        result = benchmark_int8.stats(values)
        self.assertEqual(result['samples'], 4)
        self.assertEqual(result['mean_ms'], 4)
        self.assertEqual(result['p50_ms'], 3)
        self.assertAlmostEqual(result['p95_ms'], 8.25)
        self.assertEqual(result['over_4ms'], 1)
        self.assertEqual(values, [9, 1, 4, 2])

    def test_invalid_timing_samples_rejected(self):
        for values in ([], [0], [-1], [float('nan')], [float('inf')]):
            with self.assertRaises(ValueError):
                benchmark_int8.stats(values)

    def test_empty_detections_are_not_labelled_accuracy_success(self):
        result = benchmark_int8.agreement([], [])
        self.assertIsNone(result['matched_median_iou'])
        self.assertIn('NOT ground-truth', result['note'])
        self.assertIn('does not validate', result['note'])

    def test_box_agreement_is_one_to_one(self):
        a = {'left': 0, 'top': 0, 'right': 10, 'bottom': 10}
        b = {'left': 20, 'top': 20, 'right': 30, 'bottom': 30}
        result = benchmark_int8.agreement([a], [a, a, b])
        self.assertEqual(result['matched_iou_ge_0_5'], 1)
        self.assertEqual(result['matched_median_iou'], 1)
        self.assertEqual(benchmark_int8.iou(a, b), 0)


@unittest.skipUnless(importlib.util.find_spec('onnxruntime'), 'Optional CPU quantizer not installed')
class CalibrationTests(unittest.TestCase):
    def setUp(self):
        import quantize_model
        self.quantize = quantize_model
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_rgb_letterbox_layout_and_reader_rewind(self):
        import numpy as np
        from PIL import Image
        for size, start, stop, axis in (((1280, 720), 140, 500, 2), ((320, 640), 160, 480, 3)):
            path = self.root / f'{axis}.png'
            Image.new('RGB', size, (255, 0, 0)).save(path)
            data = self.quantize.tensor(path)
            self.assertEqual(data.shape, (1, 3, 640, 640))
            self.assertEqual(data.dtype, np.float32)
            self.assertTrue(data.flags.c_contiguous)
            self.assertAlmostEqual(float(data[0, 0, 0, 0]), 114 / 255, places=6)
            content = data[:, :, start:stop, :] if axis == 2 else data[:, :, :, start:stop]
            self.assertTrue(np.all(content[:, 0] == 1))
            self.assertTrue(np.all(content[:, 1:] == 0))
            reader = self.quantize.Screenshots([path])
            self.assertIn('images', reader.get_next())
            self.assertIsNone(reader.get_next())
            reader.rewind()
            self.assertIn('images', reader.get_next())

    def arguments(self, out):
        source = self.root / 'source.onnx'
        source.write_bytes(b'user-source')
        licence = self.root / 'LICENSE.txt'
        licence.write_bytes(b'GNU AFFERO GENERAL PUBLIC LICENSE Version 3')
        image = self.root / 'image.png'
        image.write_bytes(b'never-read-in-these-tests')
        return source, ['quantize_model.py', '--model', str(source), '--licence', str(licence),
                        '--out', str(out), str(image)]

    def test_wrong_source_rejected_before_output_creation(self):
        out = self.root / 'out'
        source, argv = self.arguments(out)
        with patch.object(sys, 'argv', argv), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            self.quantize.main()
        self.assertFalse(out.exists())
        self.assertEqual(source.read_bytes(), b'user-source')

    def test_existing_output_and_source_preserved(self):
        out = self.root / 'out'
        out.mkdir()
        existing = out / 'ui_detect.int8.onnx'
        existing.write_bytes(b'user-model')
        source, argv = self.arguments(out)
        with patch.object(sys, 'argv', argv), patch.object(self.quantize, 'SOURCE_SHA256',
                hashlib.sha256(source.read_bytes()).hexdigest()), patch.object(self.quantize, 'quant_pre_process') as prepare:
            with self.assertRaises(FileExistsError):
                self.quantize.main()
        prepare.assert_not_called()
        self.assertEqual(existing.read_bytes(), b'user-model')
        self.assertEqual(source.read_bytes(), b'user-source')


@unittest.skipUnless(os.environ.get('MOTION_ENHANCER_TEST_EXE'), 'Opt-in native GPU-only fixture')
class NativePlacementTests(unittest.TestCase):
    def test_unsupported_cpu_operator_is_rejected_not_silently_run(self):
        import onnx
        from onnx import helper, TensorProto
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            nodes = [helper.make_node('Flatten', ['images'], ['flat'], axis=1),
                     helper.make_node('LabelEncoder', ['flat'], ['labelled'], domain='ai.onnx.ml',
                                      keys_floats=[0.0], values_floats=[0.0], default_float=0.0),
                     helper.make_node('Slice', ['labelled', 'start', 'end', 'axis'], ['slice']),
                     helper.make_node('Reshape', ['slice', 'shape'], ['output0'])]
            initializers = [helper.make_tensor('start', TensorProto.INT64, [1], [0]),
                            helper.make_tensor('end', TensorProto.INT64, [1], [42000]),
                            helper.make_tensor('axis', TensorProto.INT64, [1], [1]),
                            helper.make_tensor('shape', TensorProto.INT64, [3], [1, 5, 8400])]
            graph = helper.make_graph(nodes, 'intentionally-cpu-only-negative-test',
                                      [helper.make_tensor_value_info('images', TensorProto.FLOAT, [1, 3, 640, 640])],
                                      [helper.make_tensor_value_info('output0', TensorProto.FLOAT, [1, 5, 8400])],
                                      initializer=initializers)
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 19), helper.make_opsetid('ai.onnx.ml', 2)])
            model.ir_version = 10
            onnx.checker.check_model(model)
            path = folder / 'cpu-only-negative.onnx'
            onnx.save(model, path)
            image = folder / 'source.png'
            Image.new('RGB', (512, 240), (100, 100, 100)).save(image)
            output = folder / 'must-not-exist.json'
            env = dict(os.environ, MOTION_ENHANCER_UI_MODEL=str(path))
            env.pop('MOTION_ENHANCER_UI_PROFILE', None)
            result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_EXE'], '--ui-detect-test', str(image), str(output),
                                     '--adapter', 'dgpu'], env=env, capture_output=True, text=True, timeout=60)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            text = result.stdout + result.stderr
            self.assertIn('CPU EP', text)
            self.assertIn('fallback', text.lower())


if __name__ == '__main__':
    unittest.main()
