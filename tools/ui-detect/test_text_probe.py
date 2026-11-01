"""Offline-only text experiment safety/geometry tests; no model downloads."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import helper, TensorProto

import inspect_text_probe
import prepare_text_probe


class OfflineTextTests(unittest.TestCase):
    def test_blank_maps_authorize_no_regions(self):
        self.assertEqual(inspect_text_probe.decode_maps(np.zeros((1, 1, 64, 64), np.float32), 'db', .2, .4), [])
        self.assertEqual(inspect_text_probe.decode_maps(np.zeros((1, 32, 32, 2), np.float32), 'craft', .4, .7), [])

    def test_db_score_filter_and_current_contour(self):
        raw = np.zeros((1, 1, 64, 64), np.float32)
        raw[0, 0, 10:15, 20:40] = .9
        found = inspect_text_probe.decode_maps(raw, 'db', .2, .4)
        self.assertEqual(len(found), 1)
        self.assertGreater(found[0][4], .89)
        raw[0, 0, 10:15, 20:40] = .3
        self.assertEqual(inspect_text_probe.decode_maps(raw, 'db', .2, .4), [])

    def test_craft_region_support_required(self):
        raw = np.zeros((1, 32, 32, 2), np.float32)
        raw[0, 10:20, 10:20, 1] = .9  # affinity alone is not text
        self.assertEqual(inspect_text_probe.decode_maps(raw, 'craft', .4, .7), [])
        raw[0, 11:19, 11:19, 0] = .9
        self.assertEqual(len(inspect_text_probe.decode_maps(raw, 'craft', .4, .7)), 1)

    def test_letterbox_padding_and_scene_box_rejected(self):
        candidates = [(10, 10, 30, 30, .9), (0, 0, 640, 640, .9), (200, 200, 240, 220, .9)]
        regions = inspect_text_probe.source_regions(candidates, 640, 640, 2560, 1440)
        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0]['left'], 800)
        self.assertEqual(regions[0]['top'], 240)

    def test_reference_coverage_counts_union_not_duplicate_area(self):
        region = dict(left=0, top=0, right=5, bottom=10)
        self.assertEqual(inspect_text_probe.coverage([region, region], [0, 0, 10, 10]), .5)

    def test_rectangular_map_letterbox_coordinates(self):
        regions = inspect_text_probe.source_regions([(40, 22, 60, 32, .9)], (480, 272), (960, 544), 2560, 1440)
        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(regions[0]['left'], 80 * 2560 / 960)
        self.assertAlmostEqual(regions[0]['top'], 42 * 1440 / 540)

    def test_conversion_topological_sort_repairs_late_input_cast(self):
        graph = helper.make_graph([
            helper.make_node('Identity', ['cast'], ['y']),
            helper.make_node('Cast', ['x'], ['cast'], to=TensorProto.FLOAT)], 'late-cast',
            [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1])])
        model = helper.make_model(graph)
        onnx.checker.check_model(prepare_text_probe.topological_sort(model))
        self.assertEqual(model.graph.node[0].op_type, 'Cast')

    def test_bad_source_hash_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            source, out = Path(temp) / 'source.onnx', Path(temp) / 'new'
            source.write_bytes(b'not the pinned model')
            with patch.object(sys, 'argv', ['prepare_text_probe.py', '--source', str(source), '--sha256', '0' * 64,
                                           '--kind', 'db', '--out', str(out)]), self.assertRaisesRegex(ValueError, 'SHA256'):
                prepare_text_probe.main()
            self.assertFalse(out.exists())

    def test_existing_output_never_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / 'existing'
            out.mkdir()
            user = out / 'model-fp16.onnx'
            user.write_bytes(b'existing user data')
            with patch.object(sys, 'argv', ['prepare_text_probe.py', '--source', str(Path(temp) / 'unused'),
                                           '--sha256', '0' * 64, '--kind', 'db', '--out', str(out)]), \
                    self.assertRaises(FileExistsError):
                prepare_text_probe.main()
            self.assertEqual(user.read_bytes(), b'existing user data')


@unittest.skipUnless(os.environ.get('MOTION_ENHANCER_TEST_TEXT_PROBE'), 'Opt-in standalone DirectML probe')
class NativeOfflineTextTests(unittest.TestCase):
    def test_gpu_bgr_normalization_and_changed_input_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            graph = helper.make_graph([
                helper.make_node('ReduceMean', ['x'], ['y'], axes=[1], keepdims=1)], 'normalization-fixture',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [1, 3, 'h', 'w'])],
                [helper.make_tensor_value_info('y', TensorProto.FLOAT, [1, 1, 'h', 'w'])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
            model.ir_version = 10
            model = prepare_text_probe.wrap(model, 'db', 320)
            onnx.checker.check_model(model)
            source = folder / 'normalization.onnx'
            onnx.save(model, source)
            input_paths = []
            colors = ([.25, .5, .75], [.1, .2, .3])
            for i, rgb in enumerate(colors):
                data = folder / f'color-{i}.f32'
                np.broadcast_to(np.array(rgb, np.float32).reshape(1, 3, 1, 1), (1, 3, 320, 320)).copy().tofile(data)
                input_paths.append(data)
            manifest = folder / 'manifest.txt'
            manifest.write_text('\n'.join(map(str, input_paths)) + '\n')
            output = folder / 'result'
            result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_TEXT_PROBE'],
                                     os.environ['MOTION_ENHANCER_TEST_RUNTIME'], str(source), str(manifest),
                                     str(output), '4', '0'], capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors='replace'))
            for i, rgb in enumerate(colors):
                actual = np.fromfile(output / f'color-{i}.f32', np.float32)
                expected = ((np.array(rgb[::-1]) - [.485, .456, .406]) / [.229, .224, .225]).mean()
                np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)

    def test_probe_rejects_cpu_only_model_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            graph = helper.make_graph([
                helper.make_node('LabelEncoder', ['x'], ['y'], domain='ai.onnx.ml',
                                 keys_floats=[0.0], values_floats=[1.0])], 'cpu-only-negative',
                [helper.make_tensor_value_info('x', TensorProto.FLOAT, [4])],
                [helper.make_tensor_value_info('y', TensorProto.FLOAT, [4])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12), helper.make_opsetid('ai.onnx.ml', 2)])
            model.ir_version = 10
            source = folder / 'cpu-only.onnx'
            onnx.save(model, source)
            data = folder / 'input.f32'
            np.zeros(4, np.float32).tofile(data)
            manifest = folder / 'manifest.txt'
            manifest.write_text(str(data) + '\n')
            output = folder / 'must-not-succeed'
            result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_TEXT_PROBE'],
                                     os.environ['MOTION_ENHANCER_TEST_RUNTIME'], str(source), str(manifest),
                                     str(output), '4', '0'], capture_output=True, timeout=90)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((output / 'timings.json').exists())
            self.assertIn('fallback to CPU EP has been explicitly disabled',
                          (result.stdout + result.stderr).decode(errors='replace'))


if __name__ == '__main__':
    unittest.main()
