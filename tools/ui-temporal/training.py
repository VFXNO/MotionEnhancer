"""Research objective adapters only: no CLI loop, dataset builder or job launch."""
import torch

from model import TemporalUi
from mefi.train import losses as original_flow_losses
from mefi.ui_model import candidate_loss


def losses(model: TemporalUi, *, ui_frames, ui_target, ui_valid, flow_batch=None,
           occlusion=0.0, positive_weight=1.0, dice_weight=.5):
    """Scratch = unchanged original flow objective + separately detached UI loss.

    flow_batch is the ORIGINAL interpolation triplet and target, passed verbatim
    to mefi.train.losses. ui_frames is an independently labeled causal window;
    no middle-frame target is replaced. They need not be the same sample.
    """
    if ui_frames is None or ui_target is None or ui_valid is None:
        raise ValueError('UI training requires approved dense labels and causal windows; no automatic labels')
    if model.mode == 'scratch' and flow_batch is None:
        raise ValueError('Scratch mode also requires the original flow training batch/objective')
    if model.mode == 'frozen' and flow_batch is not None:
        raise ValueError('Frozen mode cannot train flow; omit flow_batch')
    # Reject label errors before doing any flow objective work.
    logits = model.training_logits(ui_frames)
    ui = candidate_loss(logits, ui_target, ui_valid,
                        positive_weight=positive_weight, dice_weight=dice_weight)
    flow = original_flow_losses(model.backbone, flow_batch, occlusion=occlusion)[0] if flow_batch is not None else ui.new_zeros(())
    return {'total': flow + ui, 'flow': flow, 'ui': ui}
