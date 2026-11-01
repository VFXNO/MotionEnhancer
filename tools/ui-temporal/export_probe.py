"""Export synthetic research fixtures only, to a NEW temporary experiment folder.

No training, downloads, active checkpoints, dataset inputs or model activation.
The trainer GUI/application cannot load this experimental multi-input contract.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

from model import FORMAT, ExportedTap, ExportedTemporal, TemporalUi
from mefi.model import Exported
import snapshot


def temporary_root():
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home() / 'AppData/Local'))) / 'Temp/opencode'


def new_output(path):
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError('Output must be a NEW experiment folder')
    if not path.is_relative_to(temporary_root().resolve()):
        raise ValueError('Research artifacts must stay under the approved LocalAppData/Temp/opencode root')
    path.mkdir(parents=True, exist_ok=False)
    return path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    if not args.untrained_fixture:
        raise ValueError('This stage exports research timing fixtures only; --untrained-fixture is required')
    if args.width < 64 or args.width % 64 or args.height < 16 or args.height % 16:
        raise ValueError('Width must be a multiple of 64 and height a multiple of 16')
    if args.without_refiner and not args.snapshot:
        raise ValueError('--without-refiner requires a pinned research --snapshot to preserve paired weights')
    # Refuse existing/active destinations before loading any optional snapshot.
    out = new_output(args.out)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    if args.snapshot:
        if not args.snapshot_sha256:
            raise ValueError('--snapshot-sha256 is required; no implicit checkpoint discovery')
        net = snapshot.load(args.snapshot, args.snapshot_sha256)
    else:
        net = TemporalUi.from_scratch(args.preset, refiner=args.refiner)
    source_arch = net.backbone.arch
    if args.without_refiner:
        net.tap.backbone = net.backbone.without_refiner()
        net.backbone.requires_grad_(net.mode == 'scratch')
    net.eval()
    # Always label these exports as untrained diagnostics, even if a future
    # approved snapshot was supplied. Export is not an activation/promotion gate.
    source = out / 'research-weights.pt'
    snapshot.save(source, net)
    frames = torch.rand(1, 6, args.height, args.width)
    state = torch.rand(1, net.state_channels, args.height // 2, args.width // 2)
    valid = torch.ones(1, 1, 1, 1)
    variants = [
        ('baseline', Exported(copy.deepcopy(net.backbone)).eval(), (frames,), ['frames'], ['flow', 'mask']),
        ('tap', ExportedTap(net.backbone), (frames,), ['frames'], ['flow', 'mask', 'features']),
        ('temporal', ExportedTemporal(net), (frames, state, valid),
         ['frames', 'previous_state', 'history_valid'], ['flow', 'mask', 'ui_allow', 'next_state'])]
    import onnx
    records = []
    references = {}
    for name, graph, inputs, in_names, out_names in variants:
        with torch.inference_mode():
            values = graph(*inputs)
        references[name] = values
        path = out / f'{name}.onnx'
        torch.onnx.export(graph, inputs, path, opset_version=18, input_names=in_names,
                          output_names=out_names, dynamo=True, optimize=True, external_data=False)
        model = onnx.load(path, load_external_data=False)
        onnx.checker.check_model(model)
        if any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer):
            raise ValueError('Research graph must embed its weights')
        if [x.name for x in model.graph.input] != in_names or [x.name for x in model.graph.output] != out_names:
            raise ValueError('Export changed the declared static I/O contract')
        shapes = {}
        for value in [*model.graph.input, *model.graph.output]:
            tensor = value.type.tensor_type
            shape = [d.dim_value for d in tensor.shape.dim]
            if tensor.elem_type != onnx.TensorProto.FLOAT or not shape or min(shape) < 1:
                raise ValueError(f'{value.name}: expected static positive FP32 external I/O')
            shapes[value.name] = shape
        records.append({'variant': name, 'path': str(path), 'sha256': digest(path),
                        'inputs': in_names, 'outputs': out_names, 'shapes': shapes})
    for candidate in ('tap', 'temporal'):
        if not all(torch.equal(a, b) for a, b in zip(references['baseline'], references[candidate][:2])):
            raise ValueError('Eager FP16 flow/blend parity failed before DirectML evaluation')
    # A: invalid startup; B/C: SAME frame input with different valid histories;
    # D: changed frame input. The native probe also restores A after buffer reuse.
    cases = [('startup', frames, torch.zeros_like(state), torch.zeros_like(valid)),
             ('history_a', frames, state, valid),
             ('history_b', frames, torch.zeros_like(state), valid),
             ('changed', torch.rand_like(frames), state, valid)]
    manifests = {name: [] for name, *_ in variants}
    inputs_out = out / 'inputs'
    inputs_out.mkdir()
    for case, f, s, v in cases:
        paths = []
        for key, data in (('frames', f), ('previous_state', s), ('history_valid', v)):
            path = inputs_out / f'{case}-{key}.f32'
            data.contiguous().numpy().tofile(path)
            paths.append(str(path))
        for name in manifests:
            selected = paths if name == 'temporal' else paths[:1]
            manifests[name].append('\t'.join([case, *selected]))
        for name, graph, _, _, names in variants:
            with torch.inference_mode():
                output = graph(f, s, v) if name == 'temporal' else graph(f)
            for key, data in zip(names, output):
                data.contiguous().numpy().tofile(out / f'reference-{name}-{case}-{key}.f32')
    for name, lines in manifests.items():
        (out / f'{name}-inputs.tsv').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    report = {'format': FORMAT, 'offline_only': True, 'deployable': False, 'untrained_fixture': True,
              'mode': net.mode, 'preset': net.backbone.preset, 'flow_arch': net.backbone.arch,
              'source_snapshot_sha256': args.snapshot_sha256 if args.snapshot else None,
              'source_flow_arch': source_arch, 'refiner_removed_for_paired_probe': args.without_refiner,
              'width': args.width, 'height': args.height, 'seed': args.seed,
              'synthetic_inputs_only': True, 'optimizer_steps_run': 0, 'training_performed': False,
              'python': sys.version, 'torch': str(torch.__version__), 'onnx': onnx.__version__,
              'weights_sha256': digest(source), 'variants': records,
              'eager_fp16_flow_blend_exact': True, 'gaming_load_tested': False,
              'accuracy_established': False, 'latency_guarantee_established': False}
    (out / 'preparation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main(argv=None):
    # PyTorch's exporter prints Unicode status marks; captured Windows output
    # otherwise defaults to cp1252 and can abort a successful graph capture.
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--from-scratch', action='store_true')
    source.add_argument('--snapshot', type=Path)
    parser.add_argument('--snapshot-sha256')
    parser.add_argument('--preset', choices=('small', 'medium', 'large'), default='medium')
    parser.add_argument('--refiner', action='store_true')
    parser.add_argument('--without-refiner', action='store_true',
                        help='Remove only the pinned snapshot refiner for the paired backbone research graph')
    parser.add_argument('--width', type=int, default=832)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--untrained-fixture', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(prepare(args), indent=2))


if __name__ == '__main__':
    main()
