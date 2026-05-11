import os
from pathlib import Path
import time
import datetime
import argparse
import json
import math
import random
import numpy as np
from ruamel.yaml import YAML
yaml = YAML(typ='safe')
from prettytable import PrettyTable

import torch
import torch.nn as nn
from typing import List
import torch.backends.cudnn as cudnn
from torch.cuda.amp import GradScaler

from transformers import BertTokenizer

import utils
from models.model_search import Search

from eval import evaluation_itm, evaluation_itc, mAP

from tta.dataset import create_test_dataset, create_test_loader, create_tta_dataset, create_tta_loader
from tta.optim import configure_tta_model, create_tta_optimizer, create_tta_scheduler
from tta.adapt import test_time_adapt_itm
from tta.utils import preprocess_tta_coefficients


def parse_eval_epochs(eval_epochs):
    if eval_epochs is None:
        return None
    if isinstance(eval_epochs, (list, tuple, set)):
        return [int(epoch) for epoch in eval_epochs]
    if isinstance(eval_epochs, int):
        return [eval_epochs]
    return [int(epoch.strip()) for epoch in str(eval_epochs).split(",") if epoch.strip()]


def find_meta_modules(model: nn.Module) -> List[str]:
    """
    Traverse a PyTorch model and return list of module names on 'meta' device.
    """
    meta_module_names = []

    for name, module in model.named_modules():
        is_meta = False

        for param in module.parameters(recurse=False):
            if param.device.type == 'meta':
                is_meta = True
                break

        if is_meta:
            meta_module_names.append(name if name else "root_model")
            continue

        for buffer in module.buffers(recurse=False):
            if buffer.device.type == 'meta':
                is_meta = True
                break

        if is_meta:
            meta_module_names.append(name if name else "root_model")

    return meta_module_names

def main(args, config):
    print('Not using distributed mode')
    args.distributed = False

    print("### Hyper-parameters:")

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("     seed:", seed)

    device = torch.device(args.device)
    print("     device:", device)

    tbs = ["epoch", "R1", "R5", "R10", "mAP", "mINP"]
    table = PrettyTable(tbs)
    for tb in tbs[1:]:
        table.custom_format[tb] = lambda f, v: f"{v:.3f}"

    print("     output_dir:", args.output_dir)

    if args.bs > 0:
        config['batch_size_tta'] = args.bs
    scheduler_config = config['scheduler']
    if args.epo > 0:
        scheduler_config['epochs'] = args.epo
        config['num_epoch'] = args.epo
    if args.lr > 0:
        config['optimizer']['lr'] = args.lr
        scheduler_config['lr'] = args.lr
        config['lr'] = args.lr
    print("     batch_size_tta:", config['batch_size_tta'])
    print("     epochs:", scheduler_config['epochs'])
    print("     lr:", scheduler_config['lr'])

    print("### Creating test dataset")
    test_dataset = create_test_dataset(config)
    print(f"     test_dataset: {len(test_dataset)}")

    print("### Creating test dataloader")
    test_loader = create_test_loader(
        [test_dataset],
        batch_size=[config['batch_size_test']],
        num_workers=[4],
        is_trains=[False],
        collate_fns=[None]
    )[0]
    print(f"     test_loader: {len(test_loader)}")

    print("### Creating model")
    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    model = Search(config=config)
    if config['load_pretrained']:
        model.load_pretrained(args.checkpoint)

    meta_list = find_meta_modules(model)
    if meta_list:
        print("[Diagnosis] Following modules are on 'meta' device:")
        for module_name in meta_list:
            print(f"  - {module_name}")
    else:
        print("[Diagnosis] All modules are on physical devices.")
    del model.text_encoder.cls.predictions

    model = model.to(device)
    print("     Total Params Sum: ", sum(p.numel() for p in model.parameters()))

    print("### Inference ITC similarity matrix")
    text_to_image_similarity, image_embeds, text_embeds, text_atts, image_feats, text_feats = evaluation_itc(
        model,
        test_loader,
        tokenizer,
        device,
        config
    )

    sims_test_result = mAP(text_to_image_similarity, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, sims_test_result['R1'], sims_test_result['R5'], sims_test_result['R10'], sims_test_result['mAP'], sims_test_result['mINP']
    ])
    print("### Zero-Shot ITC Score: ")
    print(table)

    score_test_t2i = evaluation_itm(
        model,
        device, config, args,
            text_to_image_similarity, image_embeds, text_embeds, text_atts
    )
    test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
    table.add_row([
        -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    ])
    print("### Zero-Shot ITM Score: ")
    print(table)

    if args.tta:
        print("### TTA:")

        print("### Compute ITC Uncertainty")
        recall_types, selected_query_indices, uncertainties, top1_probabilities, reciprocal_probabilities = preprocess_tta_coefficients(config, text_to_image_similarity.cpu())

        print("### Creating tta dataset")
        tta_dataset = create_tta_dataset(
            config,
            text_to_image_similarity.cpu(),
            image_embeds.cpu(),
            text_embeds.cpu(),
            text_atts.cpu(),
            recall_types,
            selected_query_indices,
            uncertainties,
            top1_probabilities,
            reciprocal_probabilities,

        )
        print(f"     tta_dataset: {len(tta_dataset)}")

        print("### Creating tta dataloader")
        tta_loader = create_tta_loader(
            [tta_dataset],
            batch_size=[config['batch_size_tta']],
            num_workers=[4],
            is_trains=[True],
            collate_fns=[None]
        )[0]
        print(f"     tta_loader: {len(tta_loader)}")

        print("### Configure adapted weights")
        model = configure_tta_model(config, model)
        print("     TTA Dropout Modules: \r\n", [(n, m, m.training) for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training] )
        print("     TTA Require Gradient Params: \r\n", [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad] )
        print("     TTA Dropout Modules Number: \r\n", sum([1 for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training]) )
        print("     TTA Require Gradient Params Sum: \r\n", sum(p.numel() for p in model.parameters() if p.requires_grad) )

        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_tta_optimizer(arg_opt, model, device)
        scheduler_config = config['scheduler']
        arg_sche = utils.AttrDict(scheduler_config)
        arg_sche['step_per_epoch'] = math.ceil( len(tta_dataset) / config['batch_size_tta'] ) * config.get('tta_steps', 1)
        lr_scheduler = create_tta_scheduler(arg_sche, optimizer)
        scaler = GradScaler()  # bf16

        print("### Start ITM Test Time Adaptation")
        start_time = time.time()
        best = 0
        best_epoch = 0
        best_logs = {}
        max_epoch = scheduler_config['epochs']
        eval_epochs = parse_eval_epochs(config.get('eval_epochs', None))
        eval_epochs = set(eval_epochs) if eval_epochs is not None else None
        for epoch in range(0, max_epoch):

            train_stats = test_time_adapt_itm(model, optimizer, scaler, epoch, device, lr_scheduler, config, tta_loader)

            if eval_epochs is not None:
                should_eval = epoch in eval_epochs
            else:
                should_eval = ((epoch+1) % 10 == 0) or (epoch+1 == max_epoch)
            if should_eval:
                score_test_t2i = evaluation_itm(
                    model,
                    device, config, args,
                    text_to_image_similarity, image_embeds, text_embeds, text_atts
                )

                test_result = mAP(score_test_t2i, test_loader.dataset.g_pids, test_loader.dataset.q_pids, table)
                table.add_row([
                    epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
                ])
                print("### TTA ITM Score: ")
                print(table)

                logs = {'epo': epoch}
                for k, v in test_result.items():
                    logs[k] = np.around(v, 3)
                for k, v in train_stats.items():
                    logs[k] = float(v)
                print('     logs: ', logs)

                for k, v in logs.items():
                    logs[k] = str(v)
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(logs) + "\n")

                result = test_result['R1']
                if result > best:
                    best = result
                    best_epoch = epoch
                    best_logs = logs

            torch.cuda.empty_cache()

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"best epoch {best_epoch} : {best_logs}")
        print(f"### best epoch {best_epoch} : {best_logs}")
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('### Time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--bs', default=0, type=int, help="mini batch size")
    parser.add_argument('--epo', default=0, type=int, help="epoch")
    parser.add_argument('--lr', default=0.0, type=float)
    parser.add_argument('--seed', default=None, type=int)
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--device', default=None)
    parser.add_argument('--eval_epochs', type=str, default=None, help="Comma-separated epoch indices, e.g. 0,1,2,4,9")
    parser.add_argument('--method', type=str, default='tcr')

    parser.add_argument('--tta_steps', type=int, default=3)
    parser.add_argument('--con_ratio', type=float, default=0.3)
    parser.add_argument('--temperature', type=float, default=0.02)
    parser.add_argument('--t', type=float, default=0.1)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    config = yaml.load(open(args.config, 'r'))
    if 'num_epoch' not in config:
        config['num_epoch'] = config['scheduler']['epochs']
    if 'lr' not in config:
        config['lr'] = config['scheduler']['lr']
    if args.seed is None:
        args.seed = int(config.get('seed', 42))
    else:
        config['seed'] = args.seed
    if args.device is None:
        args.device = config.get('device', 'cuda')
    else:
        config['device'] = args.device
    config['device'] = args.device
    if args.eval_epochs is not None:
        config['eval_epochs'] = parse_eval_epochs(args.eval_epochs)
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    main(args, config)
