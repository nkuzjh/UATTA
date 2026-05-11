from torch.optim import AdamW

from torch.optim.lr_scheduler import LambdaLR

from torch import nn

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def configure_model_irra(model, config):
    """Enable adaptation for the normalization layers used by UATTA."""
    model.train()
    model.requires_grad_(False)

    for m in model.base_model.transformer.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
    text_encoder_no_tta_layers = config.get('text_encoder_no_tta_layer', [])
    if len(text_encoder_no_tta_layers) > 0:
        for text_encoder_layer_index in text_encoder_no_tta_layers:
            text_encoder_no_tta_layer = model.base_model.transformer.resblocks[text_encoder_layer_index]
            for m in text_encoder_no_tta_layer.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                    m.requires_grad_(False)
                    m.track_running_stats = False
                    m.running_mean = None
                    m.running_var = None

    for m in model.base_model.ln_final.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

    return model

def collect_params_irra(model):
    """Collect the affine scale + shift parameters from batch norms.

    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.

    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.base_model.transformer.named_modules():

        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")

    for nm, m in model.base_model.ln_final.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_tta_model(config, model):
    model = configure_model_irra(model, config)
    if config.get("uncertainty_temper_is_learnable", False) == True:
        model.uncertainty_temper.requires_grad_(True)
    if config.get("is_prompt_learning", False) == True:
        model.requires_grad_(False)
        model.prompt_learning_embedding.requires_grad_(True)
        model.is_prompt_learning=True
    return model

def create_tta_optimizer(args, model):
    lr = args.lr
    wd = args.weight_decay
    lr_mult = getattr(args, 'lr_mult', 1)
    print("     lr: ", lr, "   lr_mult: ", lr_mult, flush=True)

    optimizer_grouped_parameters = [
        {"params": [], "weight_decay": wd, "lr": lr},
        {"params": [], "weight_decay": 0.0, "lr": lr},
        {"params": [], "weight_decay": wd, "lr": lr * lr_mult},
        {"params": [], "weight_decay": 0.0, "lr": lr * lr_mult},
    ]

    no_decay = {"bias",
                "LayerNorm.bias",
                "LayerNorm.weight",
                "norm.bias",
                "norm.weight",
                "norm1.bias",
                "norm1.weight",
                "norm2.bias",
                "norm2.weight"}

    if hasattr(model, 'init_params'):
        large_lr = model.init_params
        print("     model has 'init_params', ", len(large_lr))
    else:
        large_lr = {}

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue  # frozen weights

        if any(nd in n for nd in no_decay):
            if n in large_lr:
                optimizer_grouped_parameters[3]['params'].append(p)
            else:
                optimizer_grouped_parameters[1]['params'].append(p)
        else:  # decay
            if n in large_lr:
                optimizer_grouped_parameters[2]['params'].append(p)
            else:
                optimizer_grouped_parameters[0]['params'].append(p)

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=1e-8, betas=(0.9, 0.98))

    return optimizer

def create_tta_scheduler(args, optimizer):
    if 'num_tta_steps' not in args:
        args['num_tta_steps'] = args['epochs'] * args['step_per_epoch']
    print("     num_tta_steps: ", args['num_tta_steps'], flush=True)

    if isinstance(args['num_warmup_steps'], float):
        assert 0 <= args['num_warmup_steps'] < 1
        args['num_warmup_steps'] = int(args['num_tta_steps'] * args['num_warmup_steps'])
    print("     num_warmup_steps: ", args['num_warmup_steps'], flush=True)

    print('     sched:', args.sched, flush=True)

    if args.sched == 'linear':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            return max(
                0.0, float(args.num_tta_steps - current_step) / float(
                    max(1, args.num_tta_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    elif args.sched == 'step':
        def lr_lambda(current_step: int):
            if current_step < args.num_warmup_steps:
                return float(current_step) / float(max(1, args.num_warmup_steps))
            elif current_step < args.num_warmup_steps * 4:
                tt = 1
            elif current_step < args.num_warmup_steps * 7:
                tt = 0.5
            else:
                tt = 0.2

            return tt * max(
                0.0, float(args.num_tta_steps - current_step) / float(
                    max(1, args.num_tta_steps - args.num_warmup_steps))
            )

        lr_scheduler = LambdaLR(optimizer, lr_lambda, last_epoch=-1)

    else:
        raise NotImplementedError(f"args.sched == {args.sched}")

    return lr_scheduler
