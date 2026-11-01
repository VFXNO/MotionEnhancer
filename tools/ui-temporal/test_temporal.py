"""Synthetic CPU correctness/gradient tests; no optimizer steps or dataset I/O."""
import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model import FeatureTap, TemporalUi
from history import History, Pair
import snapshot
import training
import export_probe
from mefi.model import PyramidFlow
from mefi.train import losses as flow_losses


class SyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def setUp(self):
        torch.manual_seed(17)

    def inputs(self, model, h=64, w=96):
        return (torch.rand(1, 6, h, w), torch.rand(1, model.state_channels, (h + 1) // 2, (w + 1) // 2),
                torch.ones(1, 1, 1, 1))

    def labels(self, h=64, w=96):
        target = torch.zeros(1, 1, (h + 1) // 2, (w + 1) // 2)
        target[:, :, 5:10, 7:15] = 1
        return target, torch.ones_like(target)

    def test_tap_exact_all_presets_refinement_variants_and_odd_shapes(self):
        for preset in ('small', 'medium', 'large'):
            for refined in (False, True):
                base = PyramidFlow(preset, refiner=refined).eval()
                # Exercise nonzero flow and refiner branches, not zero-head parity only.
                with torch.no_grad():
                    for block in [base.coarse, *base.fine]:
                        block.head.weight.normal_(0, .002)
                    if refined:
                        base.fine[-1].refiner.head.weight.normal_(0, .002)
                for h, w in ((64, 96), (65, 99)):
                    with self.subTest(preset=preset, refined=refined, size=(h, w)), torch.inference_mode():
                        a, b = torch.rand(1, 3, h, w), torch.rand(1, 3, h, w)
                        expected, actual = base(a, b), FeatureTap(base)(a, b)
                        self.assertTrue(torch.equal(expected[0], actual[0]))
                        self.assertTrue(torch.equal(expected[1], actual[1]))
                        self.assertEqual(actual[2].shape[-2:], ((h + 1) // 2, (w + 1) // 2))

    def test_no_duplicate_finest_body_execution(self):
        net = TemporalUi.from_scratch('small', refiner=True).eval()
        calls = []
        hook = net.backbone.fine[-1].body.register_forward_hook(lambda *_: calls.append(1))
        try:
            with torch.inference_mode():
                net(*self.inputs(net))
            self.assertEqual(len(calls), 1)
        finally:
            hook.remove()

    def test_frozen_constructor_does_not_mutate_source_and_stays_eval(self):
        source = PyramidFlow('small', refiner=True).train()
        before = {k: v.clone() for k, v in source.state_dict().items()}
        net = TemporalUi(source, mode='frozen').train()
        self.assertTrue(source.training)
        self.assertTrue(all(p.requires_grad for p in source.parameters()))
        self.assertFalse(net.backbone.training)
        self.assertTrue(all(not p.requires_grad for p in net.backbone.parameters()))
        self.assertTrue(all(torch.equal(v, source.state_dict()[k]) for k, v in before.items()))
        groups = net.parameter_groups()
        self.assertEqual([g['name'] for g in groups], ['ui'])
        self.assertTrue({id(p) for p in groups[0]['params']}.isdisjoint({id(p) for p in net.backbone.parameters()}))

    def test_from_scratch_needs_no_checkpoint_and_both_parameter_groups(self):
        net = TemporalUi.from_scratch('medium', refiner=True)
        self.assertEqual(net.mode, 'scratch')
        self.assertTrue(all(p.requires_grad for p in net.backbone.parameters()))
        groups = net.parameter_groups()
        self.assertEqual([g['name'] for g in groups], ['flow', 'ui'])
        ids = [id(p) for g in groups for p in g['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in net.parameters()})

    def test_ui_forward_leaves_flow_and_blend_exact(self):
        net = TemporalUi.from_scratch('small', refiner=True).eval()
        frames, state, valid = self.inputs(net)
        with torch.inference_mode():
            expected = net.backbone(frames[:, :3], frames[:, 3:])
            actual = net(frames, state, valid)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(expected, actual[:2])))
        self.assertTrue(bool(((actual[2] >= 0) & (actual[2] <= 1)).all()))

    def test_invalid_history_including_nan_does_not_authorize_pixels(self):
        net = TemporalUi.from_scratch('small').eval()
        frames, state, valid = self.inputs(net)
        with torch.inference_mode():
            for flag in (0, -1, 2, float('nan')):
                result = net(frames, state, torch.full_like(valid, flag))
                self.assertEqual(torch.count_nonzero(result[2]), 0)
            state[0, 0, 0, 0] = float('nan')
            result = net(frames, state, valid)
            self.assertTrue(bool(torch.isfinite(result[2]).all()))
            self.assertEqual(torch.count_nonzero(result[2]), 0)

    def test_valid_history_is_consumed_and_first_input_restored(self):
        net = TemporalUi.from_scratch('small').eval()
        frames, state, valid = self.inputs(net)
        with torch.inference_mode():
            first = net(frames, state, valid)
            other = net(frames, torch.zeros_like(state), valid)
            restored = net(frames, state, valid)
        self.assertGreater(float((first[2] - other[2]).abs().max()), 1e-7)
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, restored)))

    def test_invalid_history_is_per_sample_and_nonfinite_logits_fail_closed(self):
        net = TemporalUi.from_scratch('small').eval()
        frames, state, valid = self.inputs(net)
        frames, state, valid = frames.repeat(2, 1, 1, 1), state.repeat(2, 1, 1, 1), valid.repeat(2, 1, 1, 1)
        state[0, 0, 0, 0] = float('inf')
        with torch.inference_mode():
            result = net(frames, state, valid)
            self.assertEqual(torch.count_nonzero(result[2][0]), 0)
            self.assertGreater(torch.count_nonzero(result[2][1]), 0)
            net.head[-1].bias.fill_(float('nan'))
            self.assertEqual(torch.count_nonzero(net(frames, state, valid)[2]), 0)

    def test_shape_contract_rejects_bad_state(self):
        net = TemporalUi.from_scratch('small')
        frames, state, valid = self.inputs(net)
        with self.assertRaisesRegex(ValueError, 'shape'):
            net(frames, state[:, :1], valid)
        with self.assertRaisesRegex(ValueError, 'frames'):
            net(frames[:, :3], state, valid)

    def test_ui_gradients_never_reach_backbone_in_either_mode(self):
        for mode in ('frozen', 'scratch'):
            with self.subTest(mode=mode):
                net = TemporalUi(PyramidFlow('small', refiner=True), mode=mode)
                target, valid = self.labels()
                before = {k: v.clone() for k, v in net.state_dict().items()}
                logits = net.training_logits(torch.rand(1, 9, 64, 96))
                training.candidate_loss(logits, target, valid).backward()
                self.assertTrue(all(p.grad is None for p in net.backbone.parameters()))
                self.assertGreater(float(net.project.weight.grad.abs().sum()), 0)
                self.assertGreater(float(net.head[-1].weight.grad.abs().sum()), 0)
                # Gradient checks are not training: no optimizer/weight updates.
                self.assertTrue(all(torch.equal(v, net.state_dict()[k]) for k, v in before.items()))

    def test_scratch_flow_loss_and_gradients_are_original_not_ui_supervision(self):
        net = TemporalUi.from_scratch('small', refiner=True)
        base = copy.deepcopy(net.backbone)
        flow_batch, ui_frames = torch.rand(1, 9, 64, 96), torch.rand(1, 9, 64, 96)
        target, valid = self.labels()
        expected = flow_losses(base, flow_batch, occlusion=0)[0]
        result = training.losses(net, flow_batch=flow_batch, ui_frames=ui_frames,
                                 ui_target=target, ui_valid=valid)
        self.assertTrue(torch.equal(expected, result['flow']))
        expected.backward()
        result['total'].backward()
        for (name, left), (other, right) in zip(base.named_parameters(), net.backbone.named_parameters()):
            self.assertEqual(name, other)
            if left.grad is None:
                self.assertIsNone(right.grad)
            else:
                self.assertTrue(torch.equal(left.grad, right.grad), name)
        self.assertGreater(float(net.head[-1].weight.grad.abs().sum()), 0)

    def test_scratch_requires_flow_and_ui_labels_and_frozen_refuses_flow(self):
        frames = torch.rand(1, 9, 64, 96)
        target, valid = self.labels()
        for mode in ('frozen', 'scratch'):
            net = TemporalUi(PyramidFlow('small'), mode=mode)
            with self.assertRaisesRegex(ValueError, 'labels'):
                training.losses(net, ui_frames=frames, ui_target=None, ui_valid=valid)
            with self.assertRaisesRegex(ValueError, 'flow'):
                training.losses(net, flow_batch=None if mode == 'scratch' else frames,
                                ui_frames=frames, ui_target=target, ui_valid=valid)

    def test_ignored_unknown_and_empty_positive_loss_rules(self):
        logits = torch.randn(1, 1, 8, 8, requires_grad=True)
        target = torch.full_like(logits, float('nan'))
        valid = torch.zeros_like(logits)
        value = training.candidate_loss(logits, target, valid)
        self.assertEqual(float(value.detach()), 0)
        value.backward()
        self.assertEqual(torch.count_nonzero(logits.grad), 0)
        self.assertTrue(bool(torch.isfinite(training.candidate_loss(
            logits, torch.zeros_like(logits), torch.ones_like(logits)))))
        with self.assertRaisesRegex(ValueError, 'coverage'):
            training.candidate_loss(logits, target, torch.ones_like(logits))
        with self.assertRaisesRegex(ValueError, 'identical'):
            training.candidate_loss(logits, target[:, :, :4], valid)

    def test_research_snapshot_roundtrip_retains_scratch_head_and_mode(self):
        for mode in ('frozen', 'scratch'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix='ui-temporal-fixture-') as temp:
                net = TemporalUi(PyramidFlow('small', refiner=True), mode=mode).eval()
                path = Path(temp) / 'research.pt'
                digest = snapshot.save(path, net)
                before = path.read_bytes()
                restored = snapshot.load(path, digest).eval()
                self.assertEqual(restored.mode, mode)
                inputs = self.inputs(net)
                with torch.inference_mode():
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(net(*inputs), restored(*inputs))))
                with self.assertRaises(FileExistsError):
                    snapshot.save(path, net)
                with self.assertRaisesRegex(ValueError, 'SHA256'):
                    snapshot.load(path, '0' * 64)
                self.assertEqual(path.read_bytes(), before)

    def test_legacy_checkpoint_cannot_silently_drop_or_add_ui_weights(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-legacy-') as temp:
            path = Path(temp) / 'legacy.pt'
            base = PyramidFlow('small')
            torch.save({'arch': base.arch, 'preset': 'small', 'model': base.state_dict()}, path)
            before = path.read_bytes()
            with self.assertRaisesRegex(ValueError, 'legacy'):
                snapshot.load(path, hashlib.sha256(before).hexdigest())
            self.assertEqual(path.read_bytes(), before)

    def test_research_snapshot_rejects_nonfinite_weights_even_with_correct_hash(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-nonfinite-') as temp:
            path = Path(temp) / 'bad-research.pt'
            net = TemporalUi.from_scratch('small')
            with torch.no_grad():
                net.project.weight[0, 0, 0, 0] = float('nan')
            digest = snapshot.save(path, net)
            with self.assertRaisesRegex(ValueError, 'Nonfinite'):
                snapshot.load(path, digest)


class HistoryTests(unittest.TestCase):
    def pair(self, a, b, **kwargs):
        return Pair(kwargs.get('generation', 1), a, b, kwargs.get('width', 96),
                    kwargs.get('height', 64), kwargs.get('variant', 'bundle-integrated'))

    def test_startup_chain_and_owned_state(self):
        history = History()
        pair = self.pair(0, 1)
        state, valid = history.prepare(pair)
        self.assertEqual(float(valid), 0)
        state.fill_(.5)
        history.commit(pair, state)
        state.zero_()
        prior, valid = history.prepare(self.pair(1, 2))
        self.assertEqual(float(valid), 1)
        self.assertTrue(bool((prior == .5).all()))
        prior.zero_()
        self.assertTrue(bool((history.prepare(self.pair(1, 2))[0] == .5).all()))

    def test_replacement_uses_matching_predecessor_not_latest_pair(self):
        history = History()
        first, second = self.pair(0, 1), self.pair(1, 2)
        state, _ = history.prepare(first)
        history.commit(first, state + .2)
        state, _ = history.prepare(second)
        history.commit(second, state + .5)
        prior, valid = history.prepare(self.pair(1, 3))
        self.assertEqual(float(valid), 1)
        self.assertTrue(bool((prior == .2).all()))
        history.commit(self.pair(1, 3), prior + .8)
        # Repeated replacement must retain the ORIGINAL matching predecessor.
        prior, valid = history.prepare(self.pair(1, 4))
        self.assertEqual(float(valid), 1)
        self.assertTrue(bool((prior == .2).all()))

    def test_gap_clears_stale_entries_and_commit_requires_prepared_pair(self):
        history = History()
        first = self.pair(0, 1)
        state, _ = history.prepare(first)
        history.commit(first, state)
        self.assertEqual(float(history.prepare(self.pair(4, 5))[1]), 0)
        self.assertEqual(history.entries, [])
        self.assertEqual(float(history.prepare(self.pair(1, 6))[1]), 0)
        with self.assertRaisesRegex(ValueError, 'unprepared'):
            history.commit(self.pair(1, 7), state)
        self.assertEqual(history.entries, [])

    def test_gap_cut_generation_size_and_variant_fail_closed(self):
        for kwargs in ({}, {'generation': 2}, {'width': 128}, {'variant': 'bundle-dis'}):
            history = History()
            pair = self.pair(0, 1)
            state, _ = history.prepare(pair)
            history.commit(pair, state)
            query = self.pair(2, 3) if not kwargs else self.pair(1, 2, **kwargs)
            self.assertEqual(float(history.prepare(query)[1]), 0)
        history = History()
        state, _ = history.prepare(self.pair(0, 1))
        history.commit(self.pair(0, 1), state)
        self.assertEqual(float(history.prepare(self.pair(1, 2), discontinuity=True)[1]), 0)

    def test_nonfinite_or_replayed_state_invalidates_history(self):
        history = History()
        pair = self.pair(0, 1)
        state, _ = history.prepare(pair)
        state[0, 0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Nonfinite|nonfinite'):
            history.commit(pair, state)
        self.assertEqual(history.entries, [])
        state, _ = history.prepare(pair)
        history.commit(pair, state)
        with self.assertRaisesRegex(ValueError, 'replayed'):
            history.prepare(pair)
        self.assertEqual(history.entries, [])


class ExportSafetyTests(unittest.TestCase):
    def test_existing_or_active_output_destination_is_refused(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-output-') as temp:
            existing = Path(temp)
            with self.assertRaises(FileExistsError):
                export_probe.new_output(existing)
            with patch.object(export_probe, 'temporary_root', return_value=existing / 'approved'):
                with self.assertRaisesRegex(ValueError, 'approved'):
                    export_probe.new_output(existing / 'not-approved')
            self.assertFalse((existing / 'not-approved').exists())

    def test_untrained_export_requires_explicit_fixture_flag(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-export-') as temp:
            destination = Path(temp) / 'must-not-exist'
            with self.assertRaisesRegex(ValueError, 'untrained-fixture'):
                export_probe.prepare(SimpleNamespace(out=destination, untrained_fixture=False))
            self.assertFalse(destination.exists())

    def test_bad_static_shape_fails_before_creating_any_output(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-shape-') as temp:
            destination = Path(temp) / 'must-not-exist'
            with self.assertRaisesRegex(ValueError, 'multiple'):
                export_probe.prepare(SimpleNamespace(out=destination, untrained_fixture=True,
                                                     width=65, height=64))
            self.assertFalse(destination.exists())

    def test_paired_bypass_requires_pinned_snapshot_before_creating_output(self):
        with tempfile.TemporaryDirectory(prefix='ui-temporal-paired-') as temp:
            destination = Path(temp) / 'must-not-exist'
            with self.assertRaisesRegex(ValueError, 'pinned research'):
                export_probe.prepare(SimpleNamespace(out=destination, untrained_fixture=True,
                                                     width=832, height=480, without_refiner=True, snapshot=None))
            self.assertFalse(destination.exists())


if __name__ == '__main__':
    unittest.main()
