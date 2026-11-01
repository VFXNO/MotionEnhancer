"""Matrix setup/conversion/numeric safety tests; no downloads, training or activation."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import onnx
from onnx import helper, TensorProto

import benchmark_matrix
import convert_matrix
import setup_matrix


class MatrixSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_existing_setup_and_provenance_preserved(self):
        existing = self.root / 'ncnn.source.json'
        existing.write_bytes(b'user provenance')
        with patch.object(sys, 'argv', ['setup_matrix.py', '--out', str(self.root / 'ncnn')]), \
                patch.object(setup_matrix.urllib.request, 'urlopen') as download, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            setup_matrix.main()
        download.assert_not_called()
        self.assertEqual(existing.read_bytes(), b'user provenance')

    def test_bad_archive_hash_writes_nothing(self):
        out = self.root / 'ncnn'
        with patch.object(sys, 'argv', ['setup_matrix.py', '--out', str(out)]), \
                patch.object(setup_matrix.urllib.request, 'urlopen', return_value=io.BytesIO(b'wrong source')), \
                self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            setup_matrix.main()
        self.assertFalse(out.exists())

    def test_path_traversal_rejected_before_creation(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('../outside.txt', 'must not write')
        data = archive.getvalue()
        out = self.root / 'ncnn'
        with patch.object(sys, 'argv', ['setup_matrix.py', '--out', str(out)]), \
                patch.object(setup_matrix.urllib.request, 'urlopen', return_value=io.BytesIO(data)), \
                patch.object(setup_matrix, 'SHA256', hashlib.sha256(data).hexdigest()), \
                self.assertRaisesRegex(RuntimeError, 'Unsafe archive path'):
            setup_matrix.main()
        self.assertFalse(out.exists())
        self.assertFalse((self.root / 'outside.txt').exists())

    def conversion_args(self, out):
        source, licence = self.root / 'source.onnx', self.root / 'LICENSE'
        model = helper.make_model(helper.make_graph([helper.make_node('Identity', ['in'], ['out'])], 'test',
            [helper.make_tensor_value_info('in', TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info('out', TensorProto.FLOAT, [1])]))
        onnx.save(model, source)
        licence.write_bytes(b'GNU AFFERO GENERAL PUBLIC LICENSE')
        return source, ['convert_matrix.py', '--model', str(source), '--licence', str(licence),
                        '--pnnx', str(self.root / 'not-run.exe'), '--out', str(out)]

    def test_wrong_model_cannot_be_converted(self):
        out = self.root / 'out'
        source, args = self.conversion_args(out)
        before = source.read_bytes()
        with patch.object(sys, 'argv', args), patch.object(convert_matrix.subprocess, 'run') as run, \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            convert_matrix.main()
        run.assert_not_called()
        self.assertFalse(out.exists())
        self.assertEqual(source.read_bytes(), before)

    def test_conversion_never_replaces_existing_output(self):
        out = self.root / 'out'
        out.mkdir()
        existing = out / 'ui_detect.matrix.bin'
        existing.write_bytes(b'user weights')
        source, args = self.conversion_args(out)
        with patch.object(sys, 'argv', args), patch.object(convert_matrix, 'SOURCE_SHA256', hashlib.sha256(source.read_bytes()).hexdigest()), \
                patch.object(convert_matrix.subprocess, 'run') as run, self.assertRaises(FileExistsError):
            convert_matrix.main()
        run.assert_not_called()
        self.assertEqual(existing.read_bytes(), b'user weights')

    def test_unpinned_converter_is_rejected_before_execution(self):
        out = self.root / 'out'
        source, args = self.conversion_args(out)
        (self.root / 'not-run.exe').write_bytes(b'not the pinned converter')
        with patch.object(sys, 'argv', args), patch.object(convert_matrix, 'SOURCE_SHA256', hashlib.sha256(source.read_bytes()).hexdigest()), \
                patch.object(convert_matrix.subprocess, 'run') as run, contextlib.redirect_stderr(io.StringIO()), \
                self.assertRaises(SystemExit):
            convert_matrix.main()
        run.assert_not_called()
        self.assertFalse(out.exists())

    def test_raw_agreement_and_nonfinite_rejection(self):
        a, b = self.root / 'a.f32', self.root / 'b.f32'
        data = np.zeros((5, 8400), dtype=np.float32)
        data[4, 10] = 0.5
        data.tofile(a)
        data[0, 10] = 1.0
        data.tofile(b)
        result = benchmark_matrix.raw_agreement(a, b)
        self.assertEqual(result['relevant_proposals'], 1)
        self.assertEqual(result['score_max_absolute_delta'], 0)
        self.assertEqual(result['relevant_box_coordinate_max_delta_detector_px'], 1)
        data[1, 2] = np.nan
        data.tofile(b)
        with self.assertRaisesRegex(RuntimeError, 'Nonfinite'):
            benchmark_matrix.raw_agreement(a, b)


@unittest.skipUnless(os.environ.get('MOTION_ENHANCER_TEST_EXE'), 'Opt-in native matrix fixtures')
class NativeMatrixSafetyTests(unittest.TestCase):
    def run_fixture(self, param_text, expected):
        from PIL import Image
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            param = folder / 'negative.param'
            param.write_text(param_text, encoding='ascii')
            param.with_suffix('.bin').write_bytes(b'')
            image, output = folder / 'input.png', folder / 'must-not-exist.json'
            Image.new('RGB', (512, 240), (100, 100, 100)).save(image)
            env = dict(os.environ, MOTION_ENHANCER_UI_BACKEND='matrix', MOTION_ENHANCER_UI_MATRIX_MODEL=str(param))
            for key in ('MOTION_ENHANCER_UI_RAW_OUTPUT', 'MOTION_ENHANCER_UI_MATRIX_TRACE', 'MOTION_ENHANCER_UI_PROFILE'):
                env.pop(key, None)
            result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_EXE'], '--ui-detect-test', str(image), str(output),
                                     '--adapter', 'dgpu'], env=env, capture_output=True, timeout=90)
            log = (result.stdout + result.stderr).decode('utf-8', errors='replace')
            self.assertNotEqual(result.returncode, 0, log)
            self.assertFalse(output.exists())
            self.assertIn(expected, log)

    def test_cpu_feature_mask_is_rejected(self):
        self.run_fixture('7767517\n2 2\nInput input 0 1 in0\nNoop noop 1 1 in0 out0 31=16\n',
                         'CPU/feature-masked layer rejected')

    def test_gpu_only_but_no_matrix_instructions_is_rejected(self):
        # 140x100x3 == 8400x5: valid output contract, but no matrix operation.
        self.run_fixture('7767517\n3 3\nInput input 0 1 in0\n'
                         'Crop crop 1 1 in0 cropped 0=0 1=0 2=0 3=140 4=100 5=3\n'
                         'Reshape reshape 1 1 cropped out0 0=8400 1=5\n', 'WMMA=0')

    @unittest.skipUnless(os.environ.get('MOTION_ENHANCER_TEST_MATRIX_MODEL'), 'Requires separate converted UI graph')
    def test_replayed_graph_consumes_new_input_and_restores_original(self):
        from PIL import Image, ImageDraw
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            a, b = folder / 'a.png', folder / 'b.png'
            Image.new('RGB', (640, 640), (30, 50, 80)).save(a)
            image = Image.new('RGB', (640, 640), (220, 230, 240))
            draw = ImageDraw.Draw(image)
            for y in range(20, 600, 40):
                draw.rectangle((20, y, 600, y + 25), fill=(60, 120, 180), outline='black', width=2)
                draw.text((30, y + 5), 'Motion Enhancer: UI matrix replay test', fill='white')
            image.save(b)

            def run(label, source, alternate=None):
                output = folder / (label + '.json')
                env = dict(os.environ, MOTION_ENHANCER_UI_BACKEND='matrix',
                           MOTION_ENHANCER_UI_MATRIX_MODEL=os.environ['MOTION_ENHANCER_TEST_MATRIX_MODEL'],
                           MOTION_ENHANCER_UI_TEST_CHECKSUM='1', MOTION_ENHANCER_UI_TEST_RUNS='12')
                for key in ('MOTION_ENHANCER_UI_TEST_ALTERNATE', 'MOTION_ENHANCER_UI_RAW_OUTPUT',
                            'MOTION_ENHANCER_UI_MATRIX_TRACE', 'MOTION_ENHANCER_UI_MATRIX_TIMING'):
                    env.pop(key, None)
                if alternate:
                    env['MOTION_ENHANCER_UI_TEST_ALTERNATE'] = str(alternate)
                result = subprocess.run([os.environ['MOTION_ENHANCER_TEST_EXE'], '--ui-detect-test', str(source),
                                        str(output), '--adapter', 'dgpu'], env=env, capture_output=True, timeout=90)
                self.assertEqual(result.returncode, 0, (result.stdout + result.stderr).decode('utf-8', errors='replace'))
                data = json.loads(output.read_text())
                self.assertGreater(data['verified_wmma_dispatches'], 0)
                self.assertEqual(data['verified_wmma_dispatches'], data['cooperative_matrix_dispatches'])
                return data['diagnostic_checksums']

            fresh_a, fresh_b = run('fresh-a', a), run('fresh-b', b)
            self.assertEqual(len(set(fresh_a)), 1)
            self.assertEqual(len(set(fresh_b)), 1)
            self.assertNotEqual(fresh_a[0], '0')
            self.assertNotEqual(fresh_a[0], fresh_b[0])
            alternating = run('alternating', a, b)
            self.assertEqual(alternating, [fresh_a[0], fresh_b[0]] * 6)


if __name__ == '__main__':
    unittest.main()
