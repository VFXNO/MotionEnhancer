"""Explicit conversion of trainer UI weights into an opt-in runtime bundle.

Does not train, read datasets, or modify the source checkpoint/flow-only model.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import uuid

import torch
from torch.nn import functional as F

from .checkpoint import save_config
from .model import ARCH, REFINED_ARCH, PRESETS, PyramidFlow, levels_for
from .ui_model import FORMAT, HEAD_ARCH, ExportedTemporal, TemporalUi


RUNTIME_FORMAT = 'mefi-pyramid-ui-1'
TRAIN_FORMAT = 'mefi-ui-training-1'
TRAIN_ARCH = 'pair-ui-training-v1'


class RuntimeTemporal(ExportedTemporal):
    """Full network-resolution allow map: aligned FP32 rows for D3D12 copies.

    The research/training head itself stays at half resolution. Flow/mask are
    unchanged, and projected state is cached only by the live runtime.
    """
    def forward(self, frames, previous_state, history_valid):
        flow, mask, allow, state = super().forward(frames, previous_state, history_valid)
        allow = F.interpolate(allow, size=frames.shape[-2:], mode='bilinear', align_corners=False)
        return flow, mask, allow, state


def load_ui_checkpoint(checkpoint):
    ck = checkpoint
    if not isinstance(ck, dict) or (ck.get('format'), ck.get('arch'), ck.get('head_format'), ck.get('head_arch')) != (
            TRAIN_FORMAT, TRAIN_ARCH, FORMAT, HEAD_ARCH):
        raise ValueError('Choose a trainer best_ui.pt or last_ui.pt, not a flow-only or research snapshot')
    if ck.get('kind') not in ('weights', 'resume') or ck.get('deployable') is not False:
        raise ValueError('Invalid UI training checkpoint metadata')
    if ck.get('flow_arch') not in (ARCH, REFINED_ARCH) or ck.get('preset') not in PRESETS:
        raise ValueError('Unsupported UI flow backbone')
    if type(ck.get('ui_updates')) is not int or ck['ui_updates'] < 1:
        raise ValueError('The UI head has no completed training updates; train it before runtime export')
    config = ck.get('config', {})
    if config.get('mode') not in ('joint', 'frozen'):
        raise ValueError('Missing UI training mode')
    for key, maximum in (('state_channels', 32), ('hidden', 64)):
        if type(ck.get(key)) is not int or not 1 <= ck[key] <= maximum:
            raise ValueError(f'Invalid UI {key}')
    net = TemporalUi(PyramidFlow(ck['preset'], refiner=ck['flow_arch'] == REFINED_ARCH),
                     mode='scratch' if config['mode'] == 'joint' else 'frozen',
                     state_channels=ck['state_channels'], hidden=ck['hidden'])
    net.load_state_dict(ck['model'], strict=True)
    if not all(bool(torch.isfinite(t).all()) and bool(torch.isfinite(t.half()).all())
               for t in net.state_dict().values()):
        raise ValueError('UI checkpoint contains nonfinite or FP16-overflowing weights')
    return net.eval()


def export_checkpoint(ck, args, emit):
    net = load_ui_checkpoint(ck)
    out = Path(args.out).resolve()
    if out.name.lower() == 'ai_flow.onnx':
        out = out.with_name('ai_flow_ui.onnx')
        emit(event='warning', message='UI bundle is exported separately as ai_flow_ui.onnx; ai_flow.onnx is unchanged.')
    if out.with_suffix('.json').is_file():
        old = json.loads(out.with_suffix('.json').read_text(encoding='utf-8'))
        if old.get('format') != RUNTIME_FORMAT:
            raise ValueError('Output belongs to a different model format. Choose a separate UI export filename.')
    if out == Path(args.checkpoint).resolve():
        raise ValueError('The output must not replace the source checkpoint')
    out.parent.mkdir(parents=True, exist_ok=True)
    refined = net.backbone.arch == REFINED_ARCH
    variants = []
    if refined:
        bypass = copy.deepcopy(net)
        bypass.tap.backbone = net.backbone.without_refiner()
        variants.append((out.with_suffix('.dis.onnx'), bypass, False))
    variants.append((out, net, refined))
    frames = torch.zeros(1, 6, args.height, args.width)
    state = torch.zeros(1, net.state_channels, args.height // 2, args.width // 2)
    valid = torch.ones(1, 1, 1, 1)
    bundle = uuid.uuid4().hex
    # Do all exports/checks before publishing anything. Metadata is published
    # last; a partial paired update cannot silently mix different bundles.
    with tempfile.TemporaryDirectory(prefix='.ui-export-', dir=out.parent) as staging:
        pending = []
        for destination, variant, integrated in variants:
            staged = Path(staging) / destination.name
            model = RuntimeTemporal(variant).eval()
            torch.onnx.export(model, (frames, state, valid), str(staged), opset_version=18,
                              input_names=['frames', 'previous_state', 'history_valid'],
                              output_names=['flow', 'mask', 'ui_allow', 'next_state'],
                              dynamo=True, optimize=True, external_data=False)
            import onnx
            graph = onnx.load(str(staged), load_external_data=False)
            onnx.checker.check_model(graph)
            expected = {'frames': [1, 6, args.height, args.width],
                        'previous_state': list(state.shape), 'history_valid': [1, 1, 1, 1],
                        'flow': [1, 4, args.height, args.width], 'mask': [1, 1, args.height, args.width],
                        'ui_allow': [1, 1, args.height, args.width], 'next_state': list(state.shape)}
            for value in (*graph.graph.input, *graph.graph.output):
                tensor = value.type.tensor_type
                if tensor.elem_type != onnx.TensorProto.FLOAT or [d.dim_value for d in tensor.shape.dim] != expected.pop(value.name, None):
                    raise ValueError(f'Unexpected UI graph tensor contract: {value.name}')
            if expected or any(t.data_location == onnx.TensorProto.EXTERNAL for t in graph.graph.initializer):
                raise ValueError('UI export must have all seven tensors and embedded weights')
            meta = {'format': RUNTIME_FORMAT, 'arch': variant.backbone.arch, 'head_format': FORMAT,
                    'head_arch': HEAD_ARCH, 'width': args.width, 'height': args.height,
                    'state_channels': net.state_channels, 'state_width': args.width // 2,
                    'state_height': args.height // 2, 'ui_threshold': .5,
                    'levels': levels_for(args.height, args.width), 'preset': ck['preset'],
                    'integrated_refiner': integrated, 'bundle_id': bundle,
                    'checkpoint': str(Path(args.checkpoint).resolve()), 'ui_updates': ck['ui_updates'],
                    'flow_updates': ck.get('flow_updates'), 'validation': ck.get('validation'),
                    'psnr': ck.get('flow_psnr') if integrated or not refined else None,
                    'experimental': True, 'onnx_sha256': hashlib.sha256(staged.read_bytes()).hexdigest(),
                    'native_evidence_required': True}
            if refined:
                meta.update(dis_model=out.with_suffix('.dis.onnx').name, refined_model=out.name)
            pending.append((staged, destination, meta))
        for staged, destination, meta in pending:
            os.replace(staged, destination)
            save_config(meta, destination.with_suffix('.json'))
            emit(event='variant', out=str(destination), integrated_refiner=meta['integrated_refiner'])
    emit(event='done', out=str(out), dis_out=str(out.with_suffix('.dis.onnx')) if refined else None,
         runtime_name='ai_flow_ui.onnx', **meta)
    return 0
