"""Icon-only runtime contract regressions; no downloads, training or activation."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.environ.get('MOTION_ENHANCER_TEST_EXE'), 'Opt-in native icon-only fixtures')
class NativeIconOnlyTests(unittest.TestCase):
    def test_removed_text_choice_fails_instead_of_falling_back(self):
        result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_EXE'], '--ui-detector', 'text'],
                                capture_output=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('Text/HUD detection has been removed', (result.stdout + result.stderr).decode(errors='replace'))

    def reject_probability_map(self, size):
        import onnx
        from onnx import helper, TensorProto
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            nodes = [helper.make_node('ReduceMean', ['rgb'], ['mean'], axes=[1], keepdims=1)]
            if size == 320:
                nodes.append(helper.make_node('MaxPool', ['mean'], ['out'], kernel_shape=[2, 2], strides=[2, 2]))
            else:
                nodes.append(helper.make_node('Identity', ['mean'], ['out']))
            graph = helper.make_graph(nodes, 'removed-text-negative',
                                      [helper.make_tensor_value_info('rgb', TensorProto.FLOAT, [1, 3, 640, 640])],
                                      [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1, 1, size, size])])
            model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 12)])
            model.ir_version = 10
            helper.set_model_props(model, {'motion_enhancer.ui.kind': 'text-db-v1'})
            onnx.checker.check_model(model)
            source, image, output = folder / 'map.onnx', folder / 'image.png', folder / 'must-not-exist.json'
            onnx.save(model, source)
            Image.new('RGB', (512, 240), (100, 100, 100)).save(image)
            env = dict(os.environ, MOTION_ENHANCER_UI_MODEL=str(source), MOTION_ENHANCER_UI_BACKEND='directml')
            for key in ('MOTION_ENHANCER_UI_PROFILE', 'MOTION_ENHANCER_UI_RAW_OUTPUT', 'MOTION_ENHANCER_UI_TEST_ALTERNATE'):
                env.pop(key, None)
            result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_EXE'], '--ui-detect-test', str(image), str(output),
                                     '--ui-detector', 'icons', '--adapter', 'dgpu'], env=env, capture_output=True, timeout=90)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(output.exists())
            self.assertIn('Text/HUD maps are not supported', (result.stdout + result.stderr).decode(errors='replace'))

    def test_full_text_map_is_rejected_even_with_old_metadata(self):
        self.reject_probability_map(640)

    def test_pooled_text_map_is_rejected_even_with_old_metadata(self):
        self.reject_probability_map(320)


if __name__ == '__main__':
    unittest.main()
