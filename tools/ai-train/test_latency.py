"""Fast stdlib-only tests for maximum-latency acceptance reporting."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_latency as benchmark


class LatencyTests(unittest.TestCase):
    def test_budget_is_a_maximum_not_a_median_and_exact_limit_is_allowed(self):
        report = benchmark.summarize([1.0, 2.0, 4.0, 4.1, 8.0], 4.0)
        self.assertEqual(report['median_ms'], 4.0)
        self.assertEqual(report['over_budget'], 2)
        self.assertFalse(report['all_samples_within_budget'])
        exact = benchmark.summarize([1.0, 4.0], 4.0)
        self.assertEqual(exact['over_budget'], 0)
        self.assertTrue(exact['all_samples_within_budget'])

    def test_invalid_budgets_are_rejected(self):
        for value in (0, -1, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                benchmark.summarize([1.0], value)

    def test_invalid_samples_are_rejected(self):
        for values in ([], [-1], [float('nan')], [float('inf')]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                benchmark.summarize(values, 4)

    def source(self, root):
        path = root / 'original.json'
        path.write_text(json.dumps({'groups': {'candidate': {'stages': {
            'pipeline_gpu_sum': {'samples_ms': [2.0, 3.0, 5.0]}}}}}), encoding='utf-8')
        return path

    def test_saved_samples_can_be_rechecked_without_gpu_training_or_source_changes(self):
        with tempfile.TemporaryDirectory(prefix='latency-budget-') as temporary:
            root = Path(temporary)
            source = self.source(root)
            before = source.read_bytes()
            output = root / 'rechecked'
            with patch('sys.argv', ['benchmark_latency.py', '--out', str(output), '--from-report', str(source),
                                    '--budget-ms', '4']), contextlib.redirect_stdout(io.StringIO()), \
                    patch.object(benchmark.subprocess, 'run', side_effect=AssertionError('No GPU command allowed')):
                benchmark.main()
            saved = json.loads((output / 'report.json').read_text())
            stats = saved['groups']['candidate']['stages']['pipeline_gpu_sum']
            self.assertEqual((stats['budget_ms'], stats['over_budget']), (4, 1))
            self.assertEqual(stats['samples_ms'], [2.0, 3.0, 5.0])
            self.assertEqual(saved['acceptance']['new_gpu_runs'], 0)
            self.assertEqual(source.read_bytes(), before)

    def test_existing_output_folder_is_not_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='latency-budget-') as temporary:
            root = Path(temporary)
            source = self.source(root)
            before = source.read_bytes()
            with patch('sys.argv', ['benchmark_latency.py', '--out', str(root), '--from-report', str(source)]), \
                    self.assertRaises(FileExistsError):
                benchmark.main()
            self.assertEqual(source.read_bytes(), before)

    def test_both_refiners_are_counterbalanced_and_samples_are_not_merged(self):
        with tempfile.TemporaryDirectory(prefix='latency-routing-') as temporary:
            root = Path(temporary)
            executable = root / 'test.exe'
            executable.write_bytes(b'test-executable')
            model = root / 'paired.onnx'
            model.write_bytes(b'model-fixture')
            seen = []

            def native(command, **kwargs):
                integrated = '--ai-refinement' in command
                seen.append(integrated)
                log = 'Flow estimator: AI pyramid flow (ONNX)\n'
                log += ('AI refinement selected: integrated AI; DIS patch passes: 0\n' if integrated else
                        'AI refinement selected: DIS; DIS patch passes: 12\n')
                for stage in ('flow_gpu', 'synthesis_gpu', 'pipeline_gpu_sum', 'pipeline_serial_wall'):
                    log += 'BENCH_JSON ' + json.dumps({'stage': stage, 'count': 1,
                        'samples_ms': [1.5 if integrated else 2.5]}) + '\n'
                return SimpleNamespace(returncode=0, stdout=log.encode(), stderr=b'')

            output = root / 'report'
            arguments = ['benchmark_latency.py', '--out', str(output), '--executable', str(executable),
                         '--models', str(model), '--inputs', str(root), '--refinement', 'both',
                         '--rounds', '2', '--samples', '1']
            with patch('sys.argv', arguments), patch.object(benchmark.subprocess, 'run', side_effect=native), \
                    contextlib.redirect_stdout(io.StringIO()):
                benchmark.main()
            self.assertEqual(seen, [False, True, True, False])
            groups = json.loads((output / 'report.json').read_text())['groups']
            self.assertEqual(len(groups), 2)
            for group in groups.values():
                stats = group['stages']['pipeline_gpu_sum']
                self.assertEqual(stats['count'], 2)
                self.assertEqual(stats['samples_ms'], [1.5, 1.5] if group['refinement'] == 'ai' else [2.5, 2.5])


if __name__ == '__main__':
    unittest.main()
