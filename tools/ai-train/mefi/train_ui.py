"""Opt-in flow+UI or frozen-backbone UI training. Never invoked by flow-only runs.

Requires explicitly labeled causal UI windows, source-disjoint train/val splits
and explicit job caps. No teachers, automatic labeling, export or activation.
"""
import argparse
import math
from pathlib import Path
import random
import time
import uuid

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import lr_factor, restore_rng, rng_state, save_checkpoint, save_config
from .data import EpochBatches, FullFrames, Triplets, load_index, sample_identity, split
from .model import ARCH, COMPATIBLE, REFINED_ARCH, PyramidFlow
from .train import emit, headline, losses as flow_losses, validate as validate_flow
from .ui_data import Labels, UiFrames, batch_at_step, sha256
from .ui_model import FORMAT, HEAD_ARCH, TemporalUi, candidate_loss


TRAIN_FORMAT = 'mefi-ui-training-1'
TRAIN_ARCH = 'pair-ui-training-v1'  # Converted only by the separate UI runtime exporter.
DEFAULTS = dict(mode='joint', preset='medium', refiner=False, epochs=60, batch=16,
                lr=1e-3, crop=256, workers=4, val=.1, pan=0., warmup=-1,
                occlusion_start=2., seed=1, ui_batch=4, ui_crop=256, ui_lr=3e-4,
                positive_weight=1., dice_weight=.5, save_every=100)


class JobStopped(Exception):
    """Internal stop signal between validation samples, never an auto-retry."""


class CheckedValidation:
    def __init__(self, dataset, check):
        self.dataset, self.check = dataset, check
        self.preview_order = dataset.preview_order

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        reason = self.check()
        if reason:
            raise JobStopped(reason)
        sample = self.dataset[index]
        reason = self.check()
        if reason:
            raise JobStopped(reason)
        return sample


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--mode', choices=('joint', 'frozen'))
    p.add_argument('--labels', type=Path, help='explicit me-ui-labels-1/2 JSON manifest')
    p.add_argument('--allow-auto-ui-labels', action='store_true', default=None,
                   help='opt in to marked machine-generated TRAINING labels; never automatic validation')
    p.add_argument('--data', nargs='+', help='original flow datasets; joint mode only')
    p.add_argument('--preset', choices=('small', 'medium', 'large'))
    p.add_argument('--refiner', action='store_true', default=None)
    p.add_argument('--init-from', type=Path, help='optional original flow checkpoint; required in frozen mode')
    p.add_argument('--init-sha256', help='required exact SHA256 whenever --init-from is used')
    p.add_argument('--resume', action='store_true', help='continue last_ui.pt, never convert an old flow run')
    p.add_argument('--max-steps', type=int, help='explicit cumulative completed-step cap; required for a new UI run')
    p.add_argument('--max-minutes', type=float, help='explicit cumulative wall-clock job cap; required for a new UI run')
    for name in ('epochs', 'batch', 'crop', 'workers', 'warmup', 'seed', 'ui_batch', 'ui_crop', 'save_every'):
        p.add_argument('--' + name.replace('_', '-'), type=int)
    for name in ('lr', 'val', 'pan', 'occlusion_start', 'ui_lr', 'positive_weight', 'dice_weight'):
        p.add_argument('--' + name.replace('_', '-'), type=float)
    p.add_argument('--stop-token', help=argparse.SUPPRESS)
    return p


def settings(args, checkpoint=None):
    """Resume restores immutable experiment settings; only explicit caps may change."""
    if checkpoint is not None:
        if not isinstance(checkpoint, dict) or (checkpoint.get('format'), checkpoint.get('arch'), checkpoint.get('kind')) != (
                TRAIN_FORMAT, TRAIN_ARCH, 'resume') or checkpoint.get('resume_version') != 1:
            raise ValueError('Not a compatible resumable UI run; flow/research checkpoints are never converted')
        required = ('model', 'opt', 'sched', 'scaler', 'rng', 'step', 'config', 'identities',
                    'steps_per_epoch', 'schedule', 'elapsed_seconds', 'best', 'ui_updates', 'flow_updates',
                    'validated_epochs', 'initial_checkpoint', 'flow_arch', 'preset', 'state_channels', 'hidden')
        if not all(name in checkpoint for name in required):
            raise ValueError('UI continuation state is incomplete')
        if (checkpoint.get('head_format'), checkpoint.get('head_arch'), checkpoint.get('deployable')) != (FORMAT, HEAD_ARCH, False):
            raise ValueError('Incompatible UI head or deployment metadata')
        if any(type(checkpoint[name]) is not int or checkpoint[name] < 0 for name in ('step', 'ui_updates', 'flow_updates', 'validated_epochs')):
            raise ValueError('Invalid saved UI counters')
        config = checkpoint['config'].copy()
        if args.allow_auto_ui_labels is not None and args.allow_auto_ui_labels != config.get('allow_auto_ui_labels', False):
            raise ValueError('UI continuation restores allow_auto_ui_labels; use a new run to change label trust')
        if not config.get('experiment_id') or checkpoint.get('experiment_id') != config['experiment_id']:
            raise ValueError('UI experiment identity is missing/inconsistent')
        for key in (*DEFAULTS, 'labels', 'data', 'init_from', 'init_sha256'):
            requested = getattr(args, key)
            if requested is not None:
                if key in ('labels', 'init_from'):
                    requested = str(requested.resolve())
                if key == 'data':
                    requested = [str(Path(p).resolve()) for p in requested]
                if requested != config.get(key):
                    raise ValueError(f'UI continuation restores {key}; use a new run for changed experiment settings')
        for cap in ('max_steps', 'max_minutes'):
            if getattr(args, cap) is not None:
                config[cap] = getattr(args, cap)
    else:
        config = {key: getattr(args, key) if getattr(args, key) is not None else default
                  for key, default in DEFAULTS.items()}
        config.update(labels=str(args.labels.resolve()) if args.labels else None,
                      data=[str(Path(p).resolve()) for p in args.data] if args.data else [],
                      init_from=str(args.init_from.resolve()) if args.init_from else None,
                      init_sha256=args.init_sha256, max_steps=args.max_steps, max_minutes=args.max_minutes,
                      allow_auto_ui_labels=bool(args.allow_auto_ui_labels))
        config['experiment_id'] = uuid.uuid4().hex
    if not config.get('labels'):
        raise ValueError('UI training needs an explicit dense --labels manifest; flow triplets are not UI truth')
    if type(config.get('allow_auto_ui_labels', False)) is not bool:
        raise ValueError('Saved automatic UI label permission must be an explicit boolean')
    if config['mode'] not in ('joint', 'frozen') or config['preset'] not in ('small', 'medium', 'large'):
        raise ValueError('Invalid UI mode or model size')
    if config['mode'] == 'joint' and not config.get('data'):
        raise ValueError('Joint flow+UI training also needs the original --data flow datasets')
    if config['mode'] == 'frozen' and (not config.get('init_from') or config.get('data')):
        raise ValueError('Frozen UI mode requires --init-from and no flow --data')
    if config.get('init_from'):
        digest = config.get('init_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Initial flow weights require a lowercase exact --init-sha256 pin')
    elif config.get('init_sha256'):
        raise ValueError('--init-sha256 requires --init-from')
    for key in ('epochs', 'batch', 'crop', 'ui_batch', 'ui_crop', 'max_steps'):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f'{key} must be a positive integer (UI runs require explicit job caps)')
    for key in ('lr', 'ui_lr', 'positive_weight', 'max_minutes'):
        if not isinstance(config.get(key), (int, float)) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f'{key} must be finite and positive (UI runs require explicit job caps)')
    if config['workers'] < 0 or config['save_every'] < 0 or not 0 < config['val'] < 1:
        raise ValueError('Invalid workers, save interval or flow validation fraction')
    if not math.isfinite(config['dice_weight']) or config['dice_weight'] < 0:
        raise ValueError('dice_weight must be finite and nonnegative')
    if not math.isfinite(config['pan']) or not 0 <= config['pan'] <= 1 or not math.isfinite(config['occlusion_start']):
        raise ValueError('Invalid flow augmentation/loss settings')
    if config['ui_crop'] < 64 or config['ui_crop'] % 16 or config['warmup'] < -1:
        raise ValueError('UI crop must be at least 64 and a multiple of 16; warmup is -1 or nonnegative')
    return config


def new_model(config):
    """Pinned source loading is CPU/weights-only; no source file is changed."""
    source = None
    if config.get('init_from'):
        path = Path(config['init_from'])
        if sha256(path) != config['init_sha256']:
            raise ValueError('Initial checkpoint SHA256 mismatch; no run files have been written')
        source = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(source, dict) or source.get('arch') not in COMPATIBLE:
            raise ValueError('UI initialization requires an original compatible flow checkpoint')
        if config['mode'] == 'frozen':
            config['preset'], config['refiner'] = source['preset'], source['arch'] == REFINED_ARCH
    base = PyramidFlow(config['preset'], refiner=config['refiner'])
    if source is not None:
        if source['preset'] != config['preset']:
            raise ValueError('Initial flow checkpoint must have the same model size')
        if source['arch'] == REFINED_ARCH and not config['refiner']:
            raise ValueError('Cannot silently discard a trained refiner; enable refinement')
        weights = source['model']
        if not all(bool(torch.isfinite(t).all()) for t in weights.values()):
            raise ValueError('Initial flow checkpoint contains nonfinite weights')
        missing = set(base.state_dict()) - set(weights)
        if any(not name.startswith('fine.3.refiner.') for name in missing):
            raise ValueError('Initial checkpoint is missing original flow parameters')
        base.load_state_dict(weights, strict=not missing)
        if sha256(path) != config['init_sha256']:
            raise ValueError('Initial checkpoint changed while loading')
    return TemporalUi(base, mode='scratch' if config['mode'] == 'joint' else 'frozen')


def objective(model, ui_batch, flow_batch=None, *, occlusion=0., positive_weight=1., dice_weight=.5):
    frames, target, valid = ui_batch
    ui = candidate_loss(model.training_logits(frames), target, valid,
                        positive_weight=positive_weight, dice_weight=dice_weight)
    if model.mode == 'scratch':
        if flow_batch is None:
            raise ValueError('Joint mode requires the unchanged original flow batch/objective')
        flow, parts, _ = flow_losses(model.backbone, flow_batch, occlusion=occlusion)
    else:
        if flow_batch is not None:
            raise ValueError('Frozen UI training must not receive a flow training batch')
        flow, parts = ui.new_zeros(()), {}
    return flow + ui, {'flow_loss': flow.detach(), 'ui_loss': ui.detach(), **parts}


def clip_groups(model):
    """UI gradients must not change flow's original 1.0 clipping scale."""
    groups = model.parameter_groups()
    return [torch.nn.utils.clip_grad_norm_(g['params'], 1.) for g in groups]


@torch.no_grad()
def validate_ui(model, dataset, device, config, stop_check=None):
    model.eval()
    loss_sum, pixels, by_source = 0., 0, {}
    for index, item in enumerate(dataset.items):
        if stop_check:
            reason = stop_check()
            if reason:
                raise JobStopped(reason)
        batch = tuple(t[None].to(device) for t in dataset[index])
        if stop_check:
            reason = stop_check()
            if reason:
                raise JobStopped(reason)
        with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = model.training_logits(batch[0])
        loss = candidate_loss(logits, batch[1], batch[2], positive_weight=config['positive_weight'],
                              dice_weight=config['dice_weight'])
        known = batch[2].bool()
        predicted, truth = torch.sigmoid(logits.float()) >= .5, batch[1] >= .5
        n = int(known.sum())
        loss_sum, pixels = loss_sum + float(loss) * n, pixels + n
        counts = by_source.setdefault(item['source'], dict(tp=0, fp=0, fn=0, known=0))
        counts['tp'] += int((predicted & truth & known).sum())
        counts['fp'] += int((predicted & ~truth & known).sum())
        counts['fn'] += int((~predicted & truth & known).sum())
        counts['known'] += n
    def metrics(counts):
        tp, fp, fn = counts['tp'], counts['fp'], counts['fn']
        return {**counts, 'precision': tp / (tp + fp) if tp + fp else None,
                'recall': tp / (tp + fn) if tp + fn else None,
                'iou': tp / (tp + fp + fn) if tp + fp + fn else None}
    total = {key: sum(value[key] for value in by_source.values()) for key in ('tp', 'fp', 'fn', 'known')}
    model.train()
    return {'loss': loss_sum / max(1, pixels), 'threshold': .5,
            'binary_coverage_threshold': .5, 'total': metrics(total),
            'by_source': {key: metrics(value) for key, value in by_source.items()}}


def weight_payload(model, config, identities, step, best, initial):
    if not all(bool(torch.isfinite(t).all()) for t in model.state_dict().values()):
        raise ValueError('Nonfinite weights: the last usable UI checkpoint has been kept')
    return {'format': TRAIN_FORMAT, 'arch': TRAIN_ARCH, 'head_format': FORMAT, 'head_arch': HEAD_ARCH,
            'kind': 'weights', 'deployable': False, 'flow_arch': model.backbone.arch,
            'preset': model.backbone.preset, 'state_channels': model.state_channels, 'hidden': model.hidden,
            'model': model.state_dict(), 'config': config.copy(), 'identities': identities,
            'step': step, 'best': best, 'ui_updates': model.ui_updates, 'flow_updates': model.flow_updates,
            'initial_checkpoint': initial, 'experiment_id': config['experiment_id']}


def check_run_folder(run, resume, token):
    if resume:
        if not (run / 'last_ui.pt').is_file():
            raise ValueError('No last_ui.pt: cannot convert or continue a flow-only/research run as UI training')
        return
    if run.exists():
        files = list(run.iterdir())
        allowed_stop = len(files) == 1 and files[0].name == 'STOP' and token and files[0].read_text(encoding='utf-8').strip() == token
        if files and not allowed_stop:
            raise ValueError('UI training needs a new or empty run folder; existing runs/checkpoints are preserved')


def run(args, config, ck):
    started = time.monotonic()
    previous_elapsed = float(ck['elapsed_seconds']) if ck else 0.
    if not math.isfinite(previous_elapsed) or previous_elapsed < 0:
        raise ValueError('Invalid saved UI job duration')
    labels = Labels(config['labels'], allow_auto_labels=config.get('allow_auto_ui_labels', False))
    ui_train, ui_val = UiFrames(labels, 'train', config['ui_crop']), UiFrames(labels, 'val')
    flow_train = flow_val = None
    identities = {'labels': labels.identity, 'flow_training': None, 'flow_validation': None}
    if config['mode'] == 'joint':
        items, missing = load_index(config['data'])
        if missing:
            raise ValueError('Joint UI run requires complete flow indices; no rebuild or automatic skipping is performed')
        labels.require_flow_disjoint(items)
        train_items, val_items = split(items, config['val'])
        if not train_items or not val_items:
            raise ValueError('Joint mode needs nonempty original flow training and validation splits')
        flow_train = Triplets(train_items, config['crop'], train=True, pan=config['pan'])
        flow_val = FullFrames(val_items)
        identities.update(flow_training=sample_identity(train_items), flow_validation=sample_identity(flow_val.items))
    dataset = flow_train if flow_train is not None else ui_train
    batch_size = config['batch'] if flow_train is not None else config['ui_batch']
    batches = EpochBatches(len(dataset), batch_size, config['seed'])
    per_epoch, target_steps = batches.steps, config['epochs'] * batches.steps
    step, best = (int(ck['step']), float(ck['best'])) if ck else (0, float('inf'))
    validated_epochs = int(ck['validated_epochs']) if ck else 0
    if step < 0 or (ck and (ck['identities'] != identities or ck['steps_per_epoch'] != per_epoch)):
        raise ValueError('UI/flow samples, labels or batch layout changed; exact continuation refused without writes')
    if (step > target_steps or validated_epochs < 0 or validated_epochs > step // per_epoch
            or step - validated_epochs * per_epoch > per_epoch):
        raise ValueError('Invalid saved UI batch/validation position')
    if ck:
        if (ck['preset'], ck['flow_arch']) != (config['preset'], REFINED_ARCH if config['refiner'] else ARCH):
            raise ValueError('UI checkpoint architecture/config mismatch')
        if ck['ui_updates'] != step or ck['flow_updates'] != (step if config['mode'] == 'joint' else 0):
            raise ValueError('UI checkpoint update counters do not match its batch position')
        if math.isnan(best) or best < 0:
            raise ValueError('Invalid saved UI selection score')
        best_file = args.run / 'best_ui.pt'
        if math.isfinite(best) and not best_file.is_file():
            raise ValueError('Saved best_ui.pt is missing; restore it or use a new run instead of losing the selected UI model')
        if best_file.is_file():
            previous = torch.load(best_file, map_location='cpu', weights_only=True)
            if not isinstance(previous, dict) or (previous.get('format'), previous.get('kind'), previous.get('experiment_id'), previous.get('identities')) != (
                    TRAIN_FORMAT, 'weights', config['experiment_id'], identities):
                raise ValueError('best_ui.pt belongs to a different UI experiment; files preserved')
            if (previous.get('arch'), previous.get('head_arch'), previous.get('flow_arch'), previous.get('preset'), previous.get('deployable')) != (
                    TRAIN_ARCH, HEAD_ARCH, ck['flow_arch'], ck['preset'], False):
                raise ValueError('best_ui.pt has incompatible model metadata; files preserved')
            if not math.isfinite(float(previous['best'])) or previous['best'] < 0:
                raise ValueError('best_ui.pt contains an invalid UI selection score')
            best = min(best, float(previous['best']))
    def elapsed():
        return previous_elapsed + time.monotonic() - started
    if ((step >= target_steps and validated_epochs >= config['epochs']) or step >= config['max_steps']
            or elapsed() >= config['max_minutes'] * 60):
        emit(event='ui_done', step=step, steps=min(target_steps, config['max_steps']), already_complete=True,
             message='UI target or cumulative job cap reached. No automatic continuation or budget extension.',
             checkpoint=str(args.run / 'last_ui.pt'))
        return 0
    random.seed(config['seed'])
    np.random.seed(config['seed'] % 2 ** 32)
    torch.manual_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    emit(event='device', name=torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU', torch=str(torch.__version__))
    if ck:
        base = PyramidFlow(ck['preset'], refiner=ck['flow_arch'] == REFINED_ARCH)
        model = TemporalUi(base, mode='scratch' if config['mode'] == 'joint' else 'frozen',
                           state_channels=ck['state_channels'], hidden=ck['hidden'])
        model.load_state_dict(ck['model'], strict=True)
        model.ui_updates, model.flow_updates = ck['ui_updates'], ck['flow_updates']
        initial = ck['initial_checkpoint']
    else:
        model = new_model(config)
        initial = {'path': config['init_from'], 'sha256': config['init_sha256']} if config['init_from'] else None
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameter_groups(config['lr'], config['ui_lr']), weight_decay=1e-4)
    warmup = config['warmup'] if config['warmup'] >= 0 else min(500, target_steps // 20)
    schedule = ck['schedule'] if ck else {'kind': 'cosine', 'steps': target_steps, 'warmup': warmup}
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(s, schedule))
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')
    if ck:
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        if scaler.is_enabled():
            scaler.load_state_dict(ck['scaler'])
        restore_rng(ck['rng'], device)
    if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('UI checkpoint contains nonfinite weights')
    # This is after all input/continuation validation. Existing nonempty runs
    # were refused, so no old config/checkpoint can be replaced by a new run.
    check_run_folder(args.run, args.resume, args.stop_token)
    args.run.mkdir(parents=True, exist_ok=True)
    save_config(config, args.run / 'ui_config.json')
    emit(event='ui_resumed' if ck else 'ui_config', config=config, step=step,
         steps=min(target_steps, config['max_steps']), best=best if math.isfinite(best) else None,
         labels=labels.identity, deployable=False)
    loader = None
    if flow_train is not None:
        loader = DataLoader(flow_train, batch_sampler=batches, num_workers=config['workers'],
                            persistent_workers=config['workers'] > 0, pin_memory=device.type == 'cuda',
                            generator=torch.Generator().manual_seed(config['seed']))
    last_path, best_path = args.run / 'last_ui.pt', args.run / 'best_ui.pt'
    stopped, reason, skipped = False, '', int(ck.get('skipped', 0)) if ck else 0

    def save_last():
        payload = weight_payload(model, config, identities, step, best, initial)
        payload.update(kind='resume', resume_version=1, opt=opt.state_dict(), sched=sched.state_dict(),
                       scaler=scaler.state_dict(), rng=rng_state(device), schedule=schedule,
                       steps_per_epoch=per_epoch, elapsed_seconds=elapsed(), skipped=skipped,
                       validated_epochs=validated_epochs)
        save_checkpoint(payload, last_path)

    def stop_reason():
        stop = args.run / 'STOP'
        if stop.is_file() and (not args.stop_token or stop.read_text(encoding='utf-8').strip() == args.stop_token):
            return 'stop requested'
        if elapsed() >= config['max_minutes'] * 60:
            return 'cumulative wall-clock job cap'
        if step >= config['max_steps']:
            return 'cumulative completed-step cap'
        return ''

    try:
        for epoch in range(validated_epochs, config['epochs']):
            model.train()
            batches.epoch, batches.start_batch = epoch, step - epoch * per_epoch
            flow_iterator = iter(loader) if loader is not None else None
            for _ in range(batches.start_batch, per_epoch):
                reason = stop_reason()
                if reason:
                    stopped = True
                    break
                ui_batch = tuple(t.to(device) for t in batch_at_step(ui_train, step, config['ui_batch'], config['seed'] + 17))
                flow_batch = next(flow_iterator).to(device, non_blocking=True) if flow_iterator is not None else None
                reason = stop_reason()
                if reason:
                    stopped = True
                    break
                occlusion = min(1., max(0., (step / target_steps - config['occlusion_start']) / .2))
                with torch.autocast('cuda', dtype=torch.float16, enabled=device.type == 'cuda'):
                    loss, parts = objective(model, ui_batch, flow_batch, occlusion=occlusion,
                                            positive_weight=config['positive_weight'], dice_weight=config['dice_weight'])
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                norms = clip_groups(model)
                if bool(torch.isfinite(loss)) and all(bool(torch.isfinite(n)) for n in norms):
                    scaler.step(opt)
                    model.ui_updates += 1
                    model.flow_updates += int(config['mode'] == 'joint')
                else:
                    skipped += 1
                    raise ValueError('Nonfinite UI/flow objective or gradient; last usable checkpoints kept, no automatic retry')
                scaler.update()
                sched.step()
                step += 1
                if step % 20 == 0 or step == 1:
                    emit(event='ui_step', step=step, steps=min(target_steps, config['max_steps']), epoch=epoch,
                         loss=float(loss.detach()), **{k: float(v) for k, v in parts.items()},
                         ui_updates=model.ui_updates, flow_updates=model.flow_updates,
                         elapsed_seconds=round(elapsed(), 3), skipped=skipped)
                reason = stop_reason()
                if reason or (config['save_every'] and step % config['save_every'] == 0):
                    save_last()
                if reason:
                    stopped = True
                    break
            if stopped:
                break
            try:
                metrics = validate_ui(model, ui_val, device, config, stop_check=stop_reason)
                flow_score = None
                if flow_val is not None:
                    scores, _ = validate_flow(model.backbone, CheckedValidation(flow_val, stop_reason), device, epoch)
                    flow_score = headline(scores)[1][0]
            except JobStopped as error:
                stopped, reason = True, str(error)
                save_last()
                break
            score = metrics['loss']
            if not math.isfinite(score):
                raise ValueError('UI validation is nonfinite; usable checkpoints preserved')
            improved = score < best
            if improved:
                best = score
                payload = weight_payload(model, config, identities, step, best, initial)
                payload.update(validation=metrics, flow_psnr=flow_score)
                save_checkpoint(payload, best_path)
            save_config({'epoch': epoch, 'step': step, 'validation': metrics, 'flow_psnr': flow_score},
                        args.run / 'ui_validation.json')
            validated_epochs = epoch + 1
            save_last()
            emit(event='ui_epoch', epoch=epoch, epochs=config['epochs'], step=step,
                 steps=min(target_steps, config['max_steps']), validation=metrics,
                 flow_psnr=flow_score, best=best, improved=improved)
        save_last()
    finally:
        # Release persistent workers when a job errors or stops; no continuation
        # process is spawned and no staging/export hook is called.
        del loader
    emit(event='ui_done', stopped=stopped, reason=reason, step=step,
         steps=min(target_steps, config['max_steps']), ui_updates=model.ui_updates,
         flow_updates=model.flow_updates, elapsed_seconds=round(elapsed(), 3),
         checkpoint=str(best_path if best_path.is_file() else last_path), deployable=False)
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    args.run = args.run.resolve()
    try:
        check_run_folder(args.run, args.resume, args.stop_token)
        ck = torch.load(args.run / 'last_ui.pt', map_location='cpu', weights_only=True) if args.resume else None
        config = settings(args, ck)
        return run(args, config, ck)
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as error:
        emit(event='error', message=f'UI training: {error}')
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
