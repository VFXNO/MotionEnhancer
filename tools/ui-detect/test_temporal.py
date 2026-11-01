"""Opt-in GPU worker regressions with changing proposal evidence, not live capture.

Tiny generated ONNX fixtures exercise admission/continuation through the real
GPU preprocessing, DirectML and asynchronous publication path. They are not
trained detectors, game-HUD accuracy tests, or replacements for runtime models.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('MOTION_ENHANCER_TEST_EXE'), 'Opt-in native temporal fixtures')
class NativeTemporalTests(unittest.TestCase):
    def replay(self, base_score, gain_score, *, starts_strong=True, capacity=False):
        import numpy as np
        import onnx
        from onnx import helper, numpy_helper, TensorProto
        from PIL import Image

        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            base = np.zeros((1, 5, 8400), dtype=np.float32)
            gain = np.zeros_like(base)
            if capacity:
                for i in range(33):
                    base[0, :, i] = [20 + (i % 8) * 80, 20 + (i // 8) * 80, 10, 10, 0.11]
                base[0, 4, 32] = 0.01
                gain[0, 4, 32] = 0.94
            else:
                base[0, :, 0] = [200, 300, 40, 30, base_score]
                gain[0, 4, 0] = gain_score
                gain[0, 0, 0] = 1.0  # current coordinates must not be frozen
                base[0, :, 1] = [500, 300, 40, 30, 0.085]  # unrelated weak texture
            initializers = [numpy_helper.from_array(base, 'base'), numpy_helper.from_array(gain, 'gain'),
                            numpy_helper.from_array(np.array([1], dtype=np.int64), 'scalar_shape')]
            nodes = [helper.make_node('ReduceMean', ['rgb'], ['mean4d'], axes=[1, 2, 3], keepdims=1),
                     helper.make_node('Reshape', ['mean4d', 'scalar_shape'], ['mean']),
                     helper.make_node('Mul', ['mean', 'gain'], ['delta']),
                     helper.make_node('Add', ['base', 'delta'], ['proposals'])]
            graph = helper.make_graph(nodes, 'temporal-proposal-fixture',
                                      [helper.make_tensor_value_info('rgb', TensorProto.FLOAT, [1, 3, 640, 640])],
                                      [helper.make_tensor_value_info('proposals', TensorProto.FLOAT, [1, 5, 8400])],
                                      initializer=initializers)
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
            model.ir_version = 10
            onnx.checker.check_model(model)
            source = folder / 'fixture.onnx'
            onnx.save(model, source)
            white, black = folder / 'white.png', folder / 'black.png'
            Image.new('RGB', (640, 640), 'white').save(white)
            Image.new('RGB', (640, 640), 'black').save(black)
            first, alternate = (white, black) if starts_strong else (black, white)
            output = folder / 'sequence.json'
            env = {k: v for k, v in os.environ.items() if not k.startswith('MOTION_ENHANCER_UI_')}
            env.update(MOTION_ENHANCER_UI_MODEL=str(source), MOTION_ENHANCER_UI_TEST_RUNS='8',
                       MOTION_ENHANCER_UI_TEST_ALTERNATE=str(alternate))
            result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_EXE'], '--ui-detect-test', str(first),
                                     str(output), '--ui-detector', 'icons', '--adapter', 'dgpu'], env=env,
                                    capture_output=True, timeout=90)
            self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode(errors='replace'))
            data = json.loads(output.read_text())
            self.assertFalse(data['cpu_fallback'])
            history = data['region_history']
            self.assertEqual([s['frame_id'] for s in history], list(range(1, 9)))
            return [s['regions'] for s in history]

    def test_threshold_dips_continue_only_the_current_matching_icon(self):
        history = self.replay(0.085, 0.025)
        self.assertEqual([len(r) for r in history], [1] * 8)
        for i, regions in enumerate(history):
            self.assertAlmostEqual(regions[0]['confidence'], 0.11 if i % 2 == 0 else 0.085, places=5)
            self.assertAlmostEqual(regions[0]['left'], 179 if i % 2 == 0 else 178, places=4)

    def test_low_confidence_cannot_start_a_new_track(self):
        history = self.replay(0.085, 0.025, starts_strong=False)
        self.assertEqual([len(r) for r in history], [0, 1, 1, 1, 1, 1, 1, 1])

    def test_missing_proposals_drop_immediately_and_do_not_linger(self):
        history = self.replay(0.0, 0.11)
        self.assertEqual([len(r) for r in history], [1, 0] * 4)

    def test_below_continuation_floor_drops_immediately(self):
        history = self.replay(0.074, 0.036)
        self.assertEqual([len(r) for r in history], [1, 0] * 4)

    def test_score_ranking_changes_do_not_evict_supported_icons(self):
        history = self.replay(0, 0, starts_strong=False, capacity=True)
        self.assertEqual([len(r) for r in history], [32] * 8)
        self.assertTrue(all(regions == history[0] for regions in history))


if __name__ == '__main__':
    unittest.main()
