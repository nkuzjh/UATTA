import os
import json
import numpy as np
import time
import datetime
from prettytable import PrettyTable

import torch
import torch.distributed as dist
import torch.nn.functional as F

import utils

@torch.no_grad()
def evaluation_itc(model, data_loader, tokenizer, device, config):
    model.eval()
    start_time = time.time()

    print('     Computing text features for evaluation')
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = config['batch_size_test_text']
    text_embeds = []
    text_atts = []
    text_feats = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=config['max_tokens'], return_tensors="pt").to(device)
        text_embed = model.get_text_embeds(text_input.input_ids, text_input.attention_mask)
        text_feat = model.text_proj(text_embed[:, 0, :])
        text_feat = F.normalize(text_feat, dim=-1)

        text_embeds.append(text_embed)
        text_atts.append(text_input.attention_mask)
        text_feats.append(text_feat)

    text_embeds = torch.cat(text_embeds, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    text_feats = torch.cat(text_feats, dim=0)

    print('     Computing image features for evaluation')
    image_embeds = []
    image_feats = []
    for image, img_id in data_loader:
        image = image.to(device)
        image_embed, _ = model.get_vision_embeds(image)
        image_feat = model.vision_proj(image_embed[:, 0, :])
        image_feat = F.normalize(image_feat, dim=-1)
        image_embeds.append(image_embed)
        image_feats.append(image_feat)

    image_embeds = torch.cat(image_embeds, dim=0)
    image_feats = torch.cat(image_feats, dim=0)

    sims_matrix = image_feats @ text_feats.t()
    sims_matrix_t2i = sims_matrix.t()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('     Computing features time {}'.format(total_time_str))

    return sims_matrix_t2i, image_embeds, text_embeds, text_atts

@torch.no_grad()
def evaluation_itm(model, device, config, args, sims_matrix, image_embeds, text_embeds, text_atts):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = '      ITM Evaluation:'
    print('     Computing matching score')
    start_time = time.time()

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    score_matrix_t2i = torch.full(sims_matrix.size(), 1000.0).to(device)
    for i, sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 100, header)):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_embeds[topk_idx].to(device)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output = model.get_cross_embeds(encoder_output, encoder_att,
                                        text_embeds=text_embeds[start + i].repeat(config['k_test'], 1, 1).to(device),
                                        text_atts=text_atts[start + i].repeat(config['k_test'], 1).to(device),)[:, 0, :]
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    min_values, _ = torch.min(score_matrix_t2i, dim=1)
    replacement_tensor = min_values.view(-1, 1).expand(-1, score_matrix_t2i.size(1))
    score_matrix_t2i[score_matrix_t2i == 1000.0] = replacement_tensor[score_matrix_t2i == 1000.0]
    score_matrix_t2i = (score_matrix_t2i - score_matrix_t2i.min()) / (score_matrix_t2i.max() - score_matrix_t2i.min())

    score_sim_t2i = sims_matrix.clone().to(device)
    score_sim_t2i = (score_sim_t2i - score_sim_t2i.min()) / (score_sim_t2i.max() - score_sim_t2i.min())
    score_matrix_t2i = score_matrix_t2i + 0.002 * score_sim_t2i

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    print('     total_time', total_time)
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('     Computing matching score time {}'.format(total_time_str))
    return score_matrix_t2i.cpu().numpy()