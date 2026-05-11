import os
import time
import os.path as op
import argparse
import datetime
import json
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import yaml
from easydict import EasyDict as edict
from prettytable import PrettyTable
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.iotools import save_train_configs
from model import build_model, build_peft_model
from datasets.bases import ImageDataset, TextDataset
from datasets.build import build_transforms, __factory

from tta.utils import preprocess_tta_coefficients
from tta.dataset import IRRA_tta_dataset, create_tta_loader
from tta.optim import configure_tta_model
from solver import build_optimizer, build_lr_scheduler


def parse_eval_epochs(eval_epochs):
    if eval_epochs is None:
        return None
    if isinstance(eval_epochs, (list, tuple, set)):
        return [int(epoch) for epoch in eval_epochs]
    if isinstance(eval_epochs, int):
        return [eval_epochs]
    return [int(epoch.strip()) for epoch in str(eval_epochs).split(",") if epoch.strip()]


@torch.enable_grad()
def adapt_one_epoch(args, config, model, tta_loader, optimizer, scaler, epoch, device, scheduler):
    model = model.train()

    start_time = time.time()

    cosine_similarity_batches = []
    entropy_values = []
    uncertainty_values = []
    loss_values = []
    lr_values = []
    for iteration, (_gallery_ids_topk, images_topk, _query_ids, captions, uncertainty, top1_probability, reciprocal_probability) in enumerate(tta_loader):
        images_topk = images_topk.reshape(-1, images_topk.size(-3), images_topk.size(-2), images_topk.size(-1)).to(device)
        captions = captions.to(device)
        uncertainty = uncertainty.to(device)#([16])
        if config.get('uncertainty', None) == 'inversed_recall_proba' and config.get('uncertainty_temper_is_learnable', False):
            top1_probability = top1_probability.to(device)
            reciprocal_probability = reciprocal_probability.to(device)

        with torch.no_grad():
            image_features = model.encode_image(images_topk)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            text_features = model.encode_text(captions)
            image_features = F.normalize(image_features, p=2, dim=1)
            text_features = F.normalize(text_features, p=2, dim=1)

            image_features = image_features.reshape(-1, config['k_tta'], image_features.size(-1))
            cosine_similarities = []
            for text_feature, image_feature_topk in zip(text_features, image_features):
                cosine_similarities.append(text_feature @ image_feature_topk.t())
            cosine_similarities = torch.stack(cosine_similarities)
            cosine_similarity_batches.append(cosine_similarities.detach().cpu().float().numpy())
            scaled_similarities = cosine_similarities / args.temperature

            if config.get('entropy_type', None) == 'sigmoid_cos_diff_mean':
                centered_similarities = scaled_similarities - scaled_similarities.mean(dim=-1, keepdim=True)
                sigmoid_scores = torch.sigmoid(centered_similarities * config.get('entropy_sigmoid_temper', 1.0))
                entropy = -(sigmoid_scores * torch.log(sigmoid_scores)).sum(-1)
            else:
                entropy = -(F.softmax(scaled_similarities, dim=-1) * F.log_softmax(scaled_similarities, dim=-1)).sum(-1)

            if config.get('uncertainty_temper_is_learnable', False):
                uncertainty_temper = model.uncertainty_temper
                if config.get('uncertainty', None) == 'inversed_recall_proba':
                    uncertainty = torch.exp((1 - (top1_probability + reciprocal_probability) / 2) * uncertainty_temper)
                elif config.get('uncertainty', None) == 'diff_div_mean':
                    uncertainty = torch.exp(torch.abs(top1_probability - reciprocal_probability) / ((top1_probability + reciprocal_probability) / 2) * uncertainty_temper)
                else:
                    raise ValueError("learnable uncertainty_temper is only supported for inversed_recall_proba and diff_div_mean")

            if config.get('uncertainty', None) is not None:
                loss = entropy / uncertainty
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

        if (iteration + 1) % args.log_period == 0:
            print(f"     Epoch[{epoch}] Iteration[{iteration + 1}/{len(tta_loader)}], entropy: {entropy.mean().item():.4f}, uncertainty: {uncertainty.mean().item():.4f}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.2e}")
            entropy_values.append(entropy.mean().item())
            uncertainty_values.append(uncertainty.mean().item())
            loss_values.append(loss.item())
            lr_values.append(optimizer.param_groups[0]["lr"])

    cosine_similarities_np = np.concatenate(cosine_similarity_batches)

    print(f"     Averaged stats: entropy_avg: {np.mean(entropy_values):.4f}, uncertainty_avg: {np.mean(uncertainty_values):.4f}, loss_avg: {np.mean(loss_values):.4f}, lr_avg: {np.mean(lr_values):.2e}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('     itm tta time {}'.format(total_time_str))
    return {
        'entropy': np.mean(entropy_values),
        'uncertainty': np.mean(uncertainty_values),
        'loss': np.mean(loss_values),
        'lr': np.mean(lr_values),
    }, cosine_similarities_np

def main(cli_args):
    with open(cli_args.config_file, 'r') as f:
        args = edict(yaml.load(f, Loader=yaml.FullLoader))

    if cli_args.seed is not None:
        args.seed = cli_args.seed
    if cli_args.device is not None:
        args.device = cli_args.device
    if cli_args.output_dir is not None:
        args.output_dir = cli_args.output_dir
    if cli_args.eval_epochs is not None:
        args.eval_epochs = parse_eval_epochs(cli_args.eval_epochs)

    save_train_configs(args.output_dir, args)
    config = vars(args)

    print('Not using distributed mode')
    args.distributed = False

    print("### Hyper-parameters:")
    print("     output_dir:", args.output_dir)

    seed = config.get('seed', 42)
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

    print("     batch_size_tta:", config['batch_size_tta'])
    print("     epochs:", config['num_epoch'])
    print("     lr:", config['lr'])

    print("### Creating test dataset")
    dataset = __factory[args.dataset_name](root=args.root_dir)
    test_transforms = build_transforms(img_size=args.img_size, aug=False, is_train=False)
    ds = dataset.test
    test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'], test_transforms)
    test_txt_set = TextDataset(ds['caption_pids'], ds['captions'], text_length=args.text_length)
    print(f"     test_txt_set: {len(test_txt_set)}    test_img_set: {len(test_img_set)}")#txt=2000 img=1000

    num_workers = args.num_workers
    test_img_loader = DataLoader(
        test_img_set,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    test_txt_loader = DataLoader(
        test_txt_set,
        batch_size=args.test_batch_size,#512
        shuffle=False,
        num_workers=num_workers
    )
    print(f"     test_txt_loader: {len(test_txt_loader)}    test_img_loader: {len(test_img_loader)}")

    print("### Creating model")
    num_classes = len(dataset.train_id_container)#train_id_container3701
    if config.get('use_peft', False):
        model = build_peft_model(args, config, num_classes=num_classes)
    else:
        model = build_model(args, num_classes=num_classes)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"        [TRAINABLE] {name} (shape: {param.shape})")

    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.ckpt_dir))
    model.to(device)
    print("     Total Params Sum: ", sum(p.numel() for p in model.parameters()))# if p.requires_grad))

    print("### Zero-Shot Score: ")
    test_result, recall1, similarity, text_features, image_features, query_ids, gallery_ids, captions, images = do_inference(model, test_img_loader, test_txt_loader)
    table.add_row([
        -999, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
    ])
    print(table)

    if args.tta:
        print("### TTA:")

        print("### Compute Cos Similarity Uncertainty")
        if config.get('compute_uncertainty_with_norm_cos_sim', False):
            text_to_image_similarity = similarity.cpu()
        else:
            text_to_image_similarity = text_features.cpu() @ image_features.t().cpu()

        topk_image_indices, recall_types, selected_query_indices, uncertainties, top1_probabilities, reciprocal_probabilities = preprocess_tta_coefficients(config, text_to_image_similarity)

        print("### Creating TTA dataset")
        is_img_aug = config.get('is_image_augmentation', False)
        test_transforms = build_transforms(img_size=args.img_size, aug=is_img_aug, is_train=is_img_aug)
        tta_dataset = IRRA_tta_dataset(
            config,
            test_transforms,
            topk_image_indices.cpu(),
            query_ids.cpu(), gallery_ids.cpu(), torch.cat(captions, dim=0).cpu(), torch.cat(images, dim=0).cpu(),
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
        if not config.get('use_peft', False):
            model = configure_tta_model(config, model)
        print("     TTA Dropout Modules: \r\n", [(n, m, m.training) for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training] )
        print("     TTA Require Gradient Params: \r\n", [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad] )
        print("     TTA Dropout Modules Number: \r\n", sum([1 for n, m in model.named_modules() if isinstance(m, torch.nn.Dropout) and m.training]) )
        print("     TTA Require Gradient Params Sum: \r\n", sum(p.numel() for p in model.parameters() if p.requires_grad) )

        optimizer = build_optimizer(args, model)
        scheduler = build_lr_scheduler(args, optimizer)

        scaler = GradScaler()  # bf16

        print(f"### Start Test Time Adaptation : num_epoch = {args.num_epoch}")
        start_time = time.time()
        best = 0
        best_epoch = 0
        best_logs = {}
        eval_epochs = parse_eval_epochs(config.get('eval_epochs', None))
        eval_epochs = set(eval_epochs) if eval_epochs is not None else None
        for epoch in range(args.num_epoch):
            train_stats, cosine_similarities_np = adapt_one_epoch(args, config, model, tta_loader, optimizer, scaler, epoch, device, scheduler)

            if eval_epochs is not None:
                should_eval = epoch in eval_epochs
            else:
                should_eval = (epoch + 1 in [10, 40, 60]) or (epoch + 1 == args.num_epoch)
            if should_eval:
                print(cosine_similarities_np.shape)
                test_result, recall1, similarity, text_features, image_features, query_ids, gallery_ids, captions, images = do_inference(model, test_img_loader, test_txt_loader)
                print("### TTA Eval Score: ")
                table.add_row([
                    epoch, test_result['R1'], test_result['R5'], test_result['R10'], test_result['mAP'], test_result['mINP']
                ])
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
    parser = argparse.ArgumentParser(description="IRRA TTA")
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--eval_epochs", type=str, default=None, help="Comma-separated epoch indices, e.g. 0,1,2,4,9")
    args = parser.parse_args()

    main(args)
