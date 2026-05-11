# UATTA

Official implementation for **Pretrain-then-Adapt: Uncertainty-Aware Test-Time Adaptation for Text-based Person Search**.

This repository contains the SOTA reproduction code for the paper's Pretrain-then-Adapt experiments:

- CLIP/IRRA-based UATTA on CUHK-PEDES, ICFG-PEDES, and RSTPReid.
- X-VLM/CMP-based UATTA on PAB.

## Requirements

Create an Anaconda environment and install the dependencies:

```bash
conda create -n uatta python=3.9 -y
conda activate uatta

# Install a PyTorch build matching your CUDA version.
pip install torch torchvision torchaudio

pip install -r requirements.txt
```

## Data And Checkpoints

Please follow the upstream dataset/checkpoint pages for access instructions and license terms.

| Resource | Official link | Notes |
| --- | --- | --- |
| UATTA paper | https://arxiv.org/abs/2604.08598 | Our paper page. |
| IRRA/HAM checkpoint | https://github.com/sssaury/HAM | HAM project/checkpoint entry. |
| CLIP tokenizer vocabulary | https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz | Required by `IRRA_UATTA/utils/simple_tokenizer.py`. |
| CUHK-PEDES | http://xiaotong.me/static/projects/person-search-language/dataset.html | Official project page. |
| ICFG-PEDES | https://github.com/zifyloo/SSAN | Official dataset/code entry. |
| RSTPReid | https://github.com/NjtechCVLab/RSTPReid-Dataset | Official dataset repository. |
| CMP/X-VLM checkpoint | https://github.com/Shuyu-XJTU/CMP | CMP/PAB official repository. |
| PAB dataset | https://github.com/Shuyu-XJTU/CMP | PAB dataset/code/checkpoint entry. |

Arrange the IRRA-side files as follows:

```text
IRRA_UATTA/
├── data/
│   └── bpe_simple_vocab_16e6.txt.gz
├── checkpoints/
│   └── HAM_checkpoint/
│       └── random100w_2HAMcaptions/
│           └── best0.pth
├── input/
│   └── images/
│       ├── CUHK-PEDES/
│       ├── ICFG-PEDES/
│       └── RSTPReid/
```

Download the CLIP tokenizer vocabulary before running the IRRA-side experiments:

```bash
cd /path/to/OpenSource_Release
mkdir -p IRRA_UATTA/data
curl -L https://github.com/openai/CLIP/raw/main/clip/bpe_simple_vocab_16e6.txt.gz \
  -o IRRA_UATTA/data/bpe_simple_vocab_16e6.txt.gz
```

Arrange the CMP-side files as follows:

```text
CMP_UATTA/
├── checkpoint/
│   ├── 16m_base_model_state_step_199999.th
│   └── bert-base-uncased/
├── data/
│   └── PAB/
│       ├── annotation/
│       └── image/
```

## SOTA Reproduction

Run each experiment separately after activating the `uatta` environment. The configs evaluate exactly the same epoch set as the retained SOTA logs.

### CUHK-PEDES

```bash
cd /path/to/OpenSource_Release/IRRA_UATTA
CUDA_VISIBLE_DEVICES=0 python tta.py \
  --config_file tta_configs/ham_cuhk_tta/uatta_config.yaml \
  --seed 42 \
  --device cuda \
  --output_dir outputs/reproduce/cuhk_pedes \
  --eval_epochs 0,1,2,4,9,14,19 \
  > logs/cuhk_reproduce.log 2>&1
```

### ICFG-PEDES

```bash
cd /path/to/OpenSource_Release/IRRA_UATTA
CUDA_VISIBLE_DEVICES=0 python tta.py \
  --config_file tta_configs/ham_icfg_tta/uatta_config.yaml \
  --seed 42 \
  --device cuda \
  --output_dir outputs/reproduce/icfg_pedes \
  --eval_epochs 0,1,2,4,9 \
  > logs/icfg_reproduce.log 2>&1
```

### RSTPReid

```bash
cd /path/to/OpenSource_Release/IRRA_UATTA
CUDA_VISIBLE_DEVICES=0 python tta.py \
  --config_file tta_configs/ham_rstp_tta/uatta_config.yaml \
  --seed 42 \
  --device cuda \
  --output_dir outputs/reproduce/rstpreid \
  --eval_epochs 0,1,2,4,9,14,19,29,39,49,59 \
  > logs/rstp_reproduce.log 2>&1
```

### PAB

```bash
cd /path/to/OpenSource_Release/CMP_UATTA
mkdir -p logs outputs/reproduce/pab
CUDA_VISIBLE_DEVICES=0 python tta.py \
  --config configs/uatta_config.yaml \
  --task pab_reproduce \
  --output_dir outputs/reproduce/pab \
  --checkpoint checkpoint/16m_base_model_state_step_199999.th \
  --seed 42 \
  --device cuda \
  --eval_epochs 0,1,2,4,9,14,19,29,39,49,59 \
  --tta \
  > logs/pab_reproduce.log 2>&1
```

## Results

The original paper/retained SOTA metrics and the latest reproduced metrics are summarized below. Full adaptation traces are recorded in `SOTA_Reproduction_Report.md`.

| Dataset | Source | Epoch | Rank@1 | Rank@5 | Rank@10 | mAP | mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUHK-PEDES | Paper/retained SOTA | 9 | 70.923 | 86.891 | 91.862 | 63.503 | 47.209 |
| CUHK-PEDES | Latest reproduction | 9 | 70.809 | 86.907 | 91.862 | 63.469 | 47.192 |
| ICFG-PEDES | Paper/retained SOTA | 9 | 62.152 | 77.318 | 82.950 | 36.110 | 5.934 |
| ICFG-PEDES | Latest reproduction | 9 | 62.006 | 77.232 | 82.784 | 35.923 | 5.821 |
| RSTPReid | Paper/retained SOTA | 49 | 61.850 | 81.050 | 88.400 | 46.304 | 22.276 |
| RSTPReid | Latest reproduction | 49 | 61.950 | 81.200 | 88.300 | 46.274 | 22.157 |
| PAB | Paper/retained SOTA | 49 | 76.138 | 98.028 | 99.090 | 86.145 | 86.145 |
| PAB | Latest reproduction | 49 | 75.784 | 98.028 | 98.989 | 86.001 | 86.001 |

Small numeric differences are expected across hardware, CUDA, and PyTorch versions.

## Acknowledgements

This release directly builds on code from:

- IRRA: https://github.com/anosorae/IRRA
- CMP: https://github.com/Shuyu-XJTU/CMP

See `THIRD_PARTY_NOTICES.md` for direct fork attribution.

We also thank the following repositories for methodological or implementation references:

- 2025-ICLR-TCR: https://github.com/XLearning-SCU/2025-ICLR-TCR
- TENT: https://github.com/DequanWang/tent
- MLLM4Text-ReID: https://github.com/WentaoTan/MLLM4Text-ReID
- HAM: https://github.com/sssaury/HAM

## Citation

```bibtex
@conference{zhang2026pretrain,
      title={Pretrain-then-Adapt: Uncertainty-Aware Test-Time Adaptation for Text-based Person Search},
      author={Jiahao Zhang and Shaofei Huang and Yaxiong Wang and Zhedong Zheng},
      year={2026},
      booktitle={SIGIR},
      doi={https://doi.org/10.1145/3805712.3809598},
      url={https://arxiv.org/abs/2604.08598},
}
```

## Contact

For questions, please contact Jiahao Zhang at yc57963@um.edu.mo.
