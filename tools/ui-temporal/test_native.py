"""Opt-in generated ONNX fixtures through the actual standalone DirectML probe."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def tensor(name, shape):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


@unittest.skipUnless(os.environ.get('MOTION_ENHANCER_TEST_TEMPORAL_PROBE'), 'Opt-in standalone DirectML probe')
class NativeTests(unittest.TestCase):
    def probe(self, model, manifest, out):
        return subprocess.run([os.environ['MOTION_ENHANCER_TEST_TEMPORAL_PROBE'],
                               os.environ['MOTION_ENHANCER_TEST_RUNTIME'], str(model), str(manifest),
                               str(out), '4', '0'], capture_output=True, timeout=90)

    def test_multiple_gpu_bound_inputs_outputs_and_history_consumption(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-native-') as temp:
            root = Path(temp)
            constants = {'axes': np.array([1, 2, 3], np.int64),
                         'flow_shape': np.array([1, 4, 16, 16], np.int64),
                         'blend_shape': np.array([1, 1, 16, 16], np.int64),
                         'ui_shape': np.array([1, 1, 8, 8], np.int64),
                         'zero': np.array([0], np.float32), 'one': np.array([1], np.float32)}
            nodes = [helper.make_node('ReduceMean', ['frames', 'axes'], ['frame_mean'], keepdims=1),
                     helper.make_node('ReduceMean', ['previous_state', 'axes'], ['state_mean'], keepdims=1),
                     helper.make_node('Add', ['frame_mean', 'state_mean'], ['logit']),
                     helper.make_node('Sigmoid', ['logit'], ['probability']),
                     helper.make_node('Equal', ['history_valid', 'one'], ['valid']),
                     helper.make_node('Where', ['valid', 'probability', 'zero'], ['gated']),
                     helper.make_node('Expand', ['gated', 'ui_shape'], ['ui_allow']),
                     helper.make_node('Expand', ['frame_mean', 'flow_shape'], ['flow']),
                     helper.make_node('Expand', ['frame_mean', 'blend_shape'], ['mask']),
                     helper.make_node('Add', ['previous_state', 'frame_mean'], ['next_state'])]
            graph = helper.make_graph(nodes, 'multi-io-plumbing-only',
                                      [tensor('frames', [1, 6, 16, 16]), tensor('previous_state', [1, 8, 8, 8]),
                                       tensor('history_valid', [1, 1, 1, 1])],
                                      [tensor('flow', [1, 4, 16, 16]), tensor('mask', [1, 1, 16, 16]),
                                       tensor('ui_allow', [1, 1, 8, 8]), tensor('next_state', [1, 8, 8, 8])],
                                      [numpy_helper.from_array(a, name) for name, a in constants.items()])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
            model.ir_version = 10
            source = root / 'control.onnx'
            onnx.checker.check_model(model)
            onnx.save(model, source)
            lines = []
            for case, prior, valid in (('startup', .5, 0), ('history_a', .5, 1), ('history_b', 0, 1)):
                paths = []
                for key, shape, value in (('frames', (1, 6, 16, 16), .25),
                                           ('state', (1, 8, 8, 8), prior), ('valid', (1, 1, 1, 1), valid)):
                    path = root / f'{case}-{key}.f32'
                    np.full(shape, value, np.float32).tofile(path)
                    paths.append(str(path))
                lines.append('\t'.join([case, *paths]))
            manifest = root / 'inputs.tsv'
            manifest.write_text('\n'.join(lines) + '\n', encoding='utf-8')
            out = root / 'result'
            done = self.probe(source, manifest, out)
            self.assertEqual(done.returncode, 0, (done.stdout + done.stderr).decode(errors='replace'))
            self.assertEqual(np.count_nonzero(np.fromfile(out / 'startup-ui_allow.f32', np.float32)), 0)
            a = np.fromfile(out / 'history_a-ui_allow.f32', np.float32)
            b = np.fromfile(out / 'history_b-ui_allow.f32', np.float32)
            np.testing.assert_allclose(a, 1 / (1 + np.exp(-.75)), atol=1e-6)
            np.testing.assert_allclose(b, 1 / (1 + np.exp(-.25)), atol=1e-6)
            report = json.loads((out / 'timings.json').read_text())
            self.assertTrue(report['completed'])
            self.assertFalse(report['cpu_fallback'])
            self.assertEqual(report['first_input_restore_max_abs_drift'], 0)

    def test_cpu_only_model_fails_instead_of_silent_fallback(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-cpu-negative-') as temp:
            root = Path(temp)
            graph = helper.make_graph([
                helper.make_node('LabelEncoder', ['x'], ['y'], domain='ai.onnx.ml',
                                 keys_floats=[0.0], values_floats=[1.0])], 'cpu-only-negative',
                [tensor('x', [4])], [tensor('y', [4])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18),
                                                           helper.make_opsetid('ai.onnx.ml', 2)])
            model.ir_version = 10
            source = root / 'cpu-only.onnx'
            onnx.save(model, source)
            data = root / 'input.f32'
            np.zeros(4, np.float32).tofile(data)
            manifest = root / 'inputs.tsv'
            manifest.write_text(f'negative\t{data}\n', encoding='utf-8')
            out = root / 'must-not-succeed'
            done = self.probe(source, manifest, out)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn('fallback to CPU EP has been explicitly disabled',
                          (done.stdout + done.stderr).decode(errors='replace'))
            self.assertFalse((out / 'timings.json').exists())


if __name__ == '__main__':
    unittest.main()
