from model import objectives
from .clip_model import Transformer, TransformerVision, QuickGELU, LayerNorm, build_CLIP_from_openai_pretrained, convert_weights
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

class IRRA(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.get('is_prompt_learning', False), args.get('prompt_learning_token_num', 1), args.pretrain_choice, args.img_size, args.stride_size)
        self.config = base_cfg #{'embed_dim': 512, 'image_resolution': (384, 128), 'vision_layers': 12, 'vision_width': 768, 'vision_patch_size': 16, 'context_length': 77, 'vocab_size': 49408, 'transformer_width': 512, 'transformer_heads': 8, 'transformer_layers': 12, 'prompt_learning_token_num': 1, 'is_prompt_learning': False, 'stride_size': 16}
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature)

        if 'id' in args.loss_names:
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
            nn.init.normal_(self.classifier.weight.data, std=0.001)
            nn.init.constant_(self.classifier.bias.data, val=0.0)

        if 'mlm' in args.loss_names:
            self.cross_attn = nn.MultiheadAttention(self.embed_dim,
                                                    self.embed_dim // 64,
                                                    batch_first=True)
            self.cross_modal_transformer = TransformerVision(width=self.embed_dim,
                                                       layers=args.cmt_depth,
                                                       heads=self.embed_dim //
                                                       64)
            scale = self.cross_modal_transformer.width**-0.5

            self.ln_pre_t = LayerNorm(self.embed_dim)
            self.ln_pre_i = LayerNorm(self.embed_dim)
            self.ln_post = LayerNorm(self.embed_dim)

            proj_std = scale * ((2 * self.cross_modal_transformer.layers)**-0.5)
            attn_std = scale
            fc_std = (2 * self.cross_modal_transformer.width)**-0.5
            for block in self.cross_modal_transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

            nn.init.normal_(self.cross_attn.in_proj_weight, std=attn_std)
            nn.init.normal_(self.cross_attn.out_proj.weight, std=proj_std)

            self.mlm_head = nn.Sequential(
                OrderedDict([('dense', nn.Linear(self.embed_dim, self.embed_dim)),
                            ('gelu', QuickGELU()),
                            ('ln', LayerNorm(self.embed_dim)),
                            ('fc', nn.Linear(self.embed_dim, args.vocab_size))]))
            nn.init.normal_(self.mlm_head.dense.weight, std=fc_std)
            nn.init.normal_(self.mlm_head.fc.weight, std=proj_std)

        self.is_prompt_learning = args.get('is_prompt_learning', False)
        self.prompt_learning_embedding = torch.nn.Parameter(torch.empty(1, args.get('prompt_learning_token_num', 1), self.embed_dim))
        nn.init.normal_(self.prompt_learning_embedding, mean=0.0, std=0.02)

    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')

    def cross_former(self, q, k, v):
        x = self.cross_attn(
                self.ln_pre_t(q),
                self.ln_pre_i(k),
                self.ln_pre_i(v),
                need_weights=False)[0]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.cross_modal_transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x)
        return x

    def encode_image(self, image):
        x = self.base_model.encode_image(image)
        return x[:, 0, :].float()

    def encode_text(self, text):
        if self.is_prompt_learning:
            prompt_learning_embedding = self.prompt_learning_embedding.expand(text.size(0), -1, -1)

            x = self.base_model.encode_text(text, self.is_prompt_learning, prompt_learning_embedding)
        else:
            x = self.base_model.encode_text(text,None,None)

        if self.is_prompt_learning:
            x = x[:, prompt_learning_embedding.size(1):, :]

        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def forward(self, batch):
        ret = dict()

        images = batch['images']
        caption_ids = batch['caption_ids']
        image_feats, text_feats = self.base_model(images, caption_ids)
        i_feats = image_feats[:, 0, :].float()
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()

        logit_scale = self.logit_scale
        ret.update({'temperature': 1 / logit_scale})

        if 'itc' in self.current_task:
            ret.update({'itc_loss':objectives.compute_itc(i_feats, t_feats, logit_scale)})

        if 'sdm' in self.current_task:
            ret.update({'sdm_loss':objectives.compute_sdm(i_feats, t_feats, batch['pids'], logit_scale)})

        if 'cmpm' in self.current_task:
            ret.update({'cmpm_loss':objectives.compute_cmpm(i_feats, t_feats, batch['pids'])})

        if 'id' in self.current_task:
            image_logits = self.classifier(i_feats.half()).float()
            text_logits = self.classifier(t_feats.half()).float()
            ret.update({'id_loss':objectives.compute_id(image_logits, text_logits, batch['pids'])*self.args.id_loss_weight})

            image_pred = torch.argmax(image_logits, dim=1)
            text_pred = torch.argmax(text_logits, dim=1)

            image_precision = (image_pred == batch['pids']).float().mean()
            text_precision = (text_pred == batch['pids']).float().mean()
            ret.update({'img_acc': image_precision})
            ret.update({'txt_acc': text_precision})

        if 'mlm' in self.current_task:
            mlm_ids = batch['mlm_ids']

            mlm_feats = self.base_model.encode_text(mlm_ids)

            x = self.cross_former(mlm_feats, image_feats, image_feats)

            x = self.mlm_head(x)  # [batch_size, text_len, num_colors]

            scores = x.float().reshape(-1, self.args.vocab_size)
            mlm_labels = batch['mlm_labels'].reshape(-1)
            ret.update({'mlm_loss': objectives.compute_mlm(scores, mlm_labels)*self.args.mlm_loss_weight})

            pred = scores.max(1)[1]
            mlm_label_idx = torch.nonzero(mlm_labels)
            acc = (pred[mlm_label_idx] == mlm_labels[mlm_label_idx]).float().mean()
            ret.update({'mlm_acc': acc})

        return ret

def build_model(args, num_classes=11003):
    model = IRRA(args, num_classes)
    convert_weights(model)
    return model

from peft import LoraConfig, get_peft_model, TaskType
from peft import  PrefixTuningConfig, PromptTuningConfig, PromptEncoderConfig #AdapterConfig,

def build_peft_model(args, config, num_classes=11003):
    model = IRRA(args, num_classes)

    print("     Applying LoRA configuration...")
    if config.get('peft_type') == "lora":
        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION, # 使用 "FEATURE_EXTRACTION" 作为通用模型的安全默认值
            r=args.get('lora_r', 8),                # LoRA 的秩 (rank)
            lora_alpha=args.get('lora_alpha', 16), # LoRA alpha

            target_modules=#["q_proj", "v_proj"],
            [
                "base_model.transformer.resblocks.6.attn.in_proj_weight",
                "base_model.transformer.resblocks.6.attn.out_proj",
                "base_model.transformer.resblocks.6.mlp.c_fc",
                "base_model.transformer.resblocks.6.mlp.c_proj",

                "base_model.transformer.resblocks.7.attn.in_proj_weight",
                "base_model.transformer.resblocks.7.attn.out_proj",
                "base_model.transformer.resblocks.7.mlp.c_fc",
                "base_model.transformer.resblocks.7.mlp.c_proj",

                "base_model.transformer.resblocks.8.attn.in_proj_weight",
                "base_model.transformer.resblocks.8.attn.out_proj",
                "base_model.transformer.resblocks.8.mlp.c_fc",
                "base_model.transformer.resblocks.8.mlp.c_proj",

                "base_model.transformer.resblocks.9.attn.in_proj_weight",
                "base_model.transformer.resblocks.9.attn.out_proj",
                "base_model.transformer.resblocks.9.mlp.c_fc",
                "base_model.transformer.resblocks.9.mlp.c_proj",

                "base_model.transformer.resblocks.10.attn.in_proj_weight",
                "base_model.transformer.resblocks.10.attn.out_proj",
                "base_model.transformer.resblocks.10.mlp.c_fc",
                "base_model.transformer.resblocks.10.mlp.c_proj",

                "base_model.transformer.resblocks.11.attn.in_proj_weight",
                "base_model.transformer.resblocks.11.attn.out_proj",
                "base_model.transformer.resblocks.11.mlp.c_fc",
                "base_model.transformer.resblocks.11.mlp.c_proj",

            ],

            lora_dropout=0.05,
        )

        model = get_peft_model(model, peft_config, autocast_adapter_dtype=False)
        model.print_trainable_parameters()
    elif config.get('peft_type') == "prefix_tuning":
        peft_config = PrefixTuningConfig(
            peft_type="PREFIX_TUNING",
            task_type=TaskType.FEATURE_EXTRACTION,#CAUSAL_LM,
            token_dim=512,
            num_layers=12,
            num_attention_heads=8,
            num_virtual_tokens=30,# 您想学习的“前缀”的长度
        )
        if not hasattr(model.base_model.transformer, 'device'):
            model.base_model.transformer.device = next(model.base_model.transformer.parameters()).device

        model.requires_grad_(False)
        model.base_model.transformer = get_peft_model(model.base_model.transformer, peft_config, autocast_adapter_dtype=False)
        for m in model.base_model.ln_final.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        for m in model.base_model.transformer.ln_final.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        model.base_model.transformer.print_trainable_parameters()
    elif config.get('peft_type') == "prompt_tuning":
        peft_config = PromptTuningConfig(
            task_type=TaskType.FEATURE_EXTRACTION,#CAUSAL_LM,
            token_dim=512,
            num_layers=12,
            num_attention_heads=8,
            num_virtual_tokens=20, # 您希望学习的“软提示”的长度
        )
        if not hasattr(model.base_model.transformer, 'device'):
            model.base_model.transformer.device = next(model.base_model.transformer.parameters()).device

        model.requires_grad_(False)
        model.base_model.transformer = get_peft_model(model.base_model.transformer, peft_config, autocast_adapter_dtype=False)
        for m in model.base_model.ln_final.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
        for m in model.base_model.transformer.ln_final.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.LayerNorm):
                m.requires_grad_(True)
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

        model.base_model.transformer.print_trainable_parameters()
    else:
        raise ValueError(f"不支持的 PEFT 策略: {config.get('peft_type')}")

    return model