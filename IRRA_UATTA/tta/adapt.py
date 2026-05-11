import torch
from torch.nn import functional as F

import utils

import time
import datetime

@torch.enable_grad()
def test_time_adapt_itm(model, optimizer, scaler, epoch, device, scheduler, config, dataloader):
    model.train()

    start_time = time.time()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('entropy', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = '      TTA Epoch: [{}]'.format(epoch)
    print_freq = 100

    for iteration, (encoder_output, encoder_att, text_embeds, text_atts, uncertainty, top1_probability, reciprocal_probability) in enumerate(metric_logger.log_every(dataloader, print_freq, header)):

        encoder_output = encoder_output.reshape(-1, encoder_output.size(-2), encoder_output.size(-1)).to(device)
        encoder_att = encoder_att.reshape(-1, encoder_att.size(-1)).to(device)
        text_embeds = text_embeds.reshape(-1, text_embeds.size(-2), text_embeds.size(-1)).to(device)
        text_atts = text_atts.reshape(-1, text_atts.size(-1)).to(device)
        uncertainty = uncertainty.to(device)
        if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
            top1_probability = top1_probability.to(device)
            reciprocal_probability = reciprocal_probability.to(device)

        with torch.cuda.amp.autocast(enabled=True):
            output = model.get_cross_embeds(
                encoder_output,
                encoder_att,
                text_embeds=text_embeds,
                text_atts=text_atts
            )[:, 0, :]
            logits = model.itm_head(output)
            logits = logits.reshape(-1, config['k_tta'], 2)
            score = logits[..., 1]
            entropy = -(F.softmax(score * config['score_temper'], dim=-1) * F.log_softmax(score * config['score_temper'], dim=-1)).sum(-1)
            if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
                uncertainty_temper = model.uncertainty_temper
                uncertainty = torch.exp((1 - (top1_probability + reciprocal_probability) / 2) * uncertainty_temper)
            if config.get('uncertainty', None) is not None:
                loss = entropy / uncertainty + uncertainty
            else:
                loss = entropy
            loss = loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()
        skip_lr_sched = (scale > scaler.get_scale())
        if not skip_lr_sched:
            scheduler.step()
        optimizer.zero_grad()

        metric_logger.update(entropy=entropy.mean().item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    metric_logger.synchronize_between_processes()
    print("     Averaged stats:", metric_logger.global_avg())

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('     itm tta time {}'.format(total_time_str))
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}
