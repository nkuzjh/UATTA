# SOTA Reproduction Report

## Scope

This report records the latest SOTA-only reproduction runs under `OpenSource_Release`. No original `CMP` or `IRRA` project files were modified.

Run tag: `reproduce`

Environment: conda env `uatta`, CUDA GPU 0.

The retained logs have been sanitized for release: user-specific absolute paths, original private workspace paths, and old internal experiment names were replaced with placeholders while preserving the metric traces.

## Commands

The four experiments were executed separately using the commands documented in `README.md`.

| Experiment | Code path | Reference log | Reproduction log | Run output directory |
| --- | --- | --- | --- | --- |
| CUHK-PEDES | `IRRA_UATTA` | `IRRA_UATTA/logs/cuhk_sota.log` | `IRRA_UATTA/logs/cuhk_reproduce.log` | `IRRA_UATTA/outputs/reproduce/cuhk_pedes` |
| ICFG-PEDES | `IRRA_UATTA` | `IRRA_UATTA/logs/icfg_sota.log` | `IRRA_UATTA/logs/icfg_reproduce.log` | `IRRA_UATTA/outputs/reproduce/icfg_pedes` |
| RSTPReid | `IRRA_UATTA` | `IRRA_UATTA/logs/rstp_sota.log` | `IRRA_UATTA/logs/rstp_reproduce.log` | `IRRA_UATTA/outputs/reproduce/rstpreid` |
| PAB | `CMP_UATTA` | `CMP_UATTA/logs/pab_sota.log` | `CMP_UATTA/logs/pab_reproduce.log` | `CMP_UATTA/outputs/reproduce/pab` |

## Result Summary

The latest reproduction evaluated every epoch that appears in the retained SOTA logs.

| Experiment | Eval epochs | Reproduced best epoch | Best R1 | Best R5 | Best R10 | Best mAP | Best mINP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUHK-PEDES | 0, 1, 2, 4, 9, 14, 19 | 14 | 70.825 | 86.940 | 91.862 | 63.479 | 47.219 |
| ICFG-PEDES | 0, 1, 2, 4, 9 | 4 | 62.042 | 77.262 | 82.900 | 36.000 | 5.917 |
| RSTPReid | 0, 1, 2, 4, 9, 14, 19, 29, 39, 49, 59 | 49 | 61.950 | 81.200 | 88.300 | 46.274 | 22.157 |
| PAB | 0, 1, 2, 4, 9, 14, 19, 29, 39, 49, 59 | 29 | 75.885 | 97.927 | 99.039 | 86.046 | 86.046 |

## Per-Epoch Reproduction Traces

### CUHK-PEDES

| Epoch | R1 | R5 | R10 | mAP | mINP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -999 | 70.630 | 86.891 | 91.813 | 63.397 | 47.052 |
| 0 | 70.598 | 86.875 | 91.813 | 63.410 | 47.090 |
| 1 | 70.598 | 86.923 | 91.845 | 63.417 | 47.120 |
| 2 | 70.695 | 86.875 | 91.829 | 63.434 | 47.133 |
| 4 | 70.776 | 86.842 | 91.813 | 63.468 | 47.187 |
| 9 | 70.809 | 86.907 | 91.862 | 63.469 | 47.192 |
| 14 | 70.825 | 86.940 | 91.862 | 63.479 | 47.219 |
| 19 | 70.825 | 86.891 | 91.862 | 63.474 | 47.249 |

### ICFG-PEDES

| Epoch | R1 | R5 | R10 | mAP | mINP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -999 | 60.636 | 77.529 | 83.273 | 35.543 | 6.005 |
| 0 | 61.921 | 77.479 | 82.966 | 36.056 | 5.921 |
| 1 | 61.926 | 77.267 | 82.971 | 36.021 | 5.959 |
| 2 | 61.971 | 77.257 | 82.905 | 36.046 | 5.963 |
| 4 | 62.042 | 77.262 | 82.900 | 36.000 | 5.917 |
| 9 | 62.006 | 77.232 | 82.784 | 35.923 | 5.821 |

### RSTPReid

| Epoch | R1 | R5 | R10 | mAP | mINP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -999 | 59.600 | 80.050 | 87.100 | 44.128 | 20.551 |
| 0 | 60.900 | 80.650 | 88.050 | 45.367 | 21.512 |
| 1 | 61.000 | 80.550 | 88.100 | 45.441 | 21.565 |
| 2 | 61.100 | 81.250 | 88.450 | 45.747 | 21.813 |
| 4 | 61.400 | 81.050 | 88.500 | 45.872 | 21.920 |
| 9 | 60.950 | 81.000 | 88.400 | 46.091 | 22.157 |
| 14 | 60.950 | 80.700 | 88.450 | 46.014 | 22.097 |
| 19 | 61.000 | 80.800 | 88.250 | 46.099 | 22.110 |
| 29 | 61.500 | 81.200 | 88.300 | 46.220 | 22.199 |
| 39 | 61.650 | 81.300 | 88.550 | 46.283 | 22.218 |
| 49 | 61.950 | 81.200 | 88.300 | 46.274 | 22.157 |
| 59 | 61.700 | 81.650 | 88.150 | 46.428 | 22.288 |

### PAB

| Epoch | R1 | R5 | R10 | mAP | mINP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| -999 | 53.438 | 86.855 | 92.922 | 68.348 | 68.348 |
| -999 | 72.700 | 97.776 | 99.090 | 84.322 | 84.322 |
| 0 | 72.801 | 97.927 | 99.090 | 84.395 | 84.395 |
| 1 | 73.155 | 97.978 | 99.090 | 84.622 | 84.622 |
| 2 | 73.610 | 97.927 | 99.039 | 84.849 | 84.849 |
| 4 | 74.419 | 97.877 | 99.039 | 85.278 | 85.278 |
| 9 | 74.975 | 97.978 | 98.989 | 85.607 | 85.607 |
| 14 | 75.683 | 98.079 | 99.090 | 86.038 | 86.038 |
| 19 | 75.379 | 97.978 | 99.090 | 85.851 | 85.851 |
| 29 | 75.885 | 97.927 | 99.039 | 86.046 | 86.046 |
| 39 | 75.834 | 98.028 | 99.039 | 86.018 | 86.018 |
| 49 | 75.784 | 98.028 | 98.989 | 86.001 | 86.001 |
| 59 | 75.784 | 98.028 | 98.989 | 86.010 | 86.010 |

## Reference Comparison

The reproduced metrics are within the expected stochastic variation range of the retained SOTA runs.

| Experiment | SOTA best epoch | SOTA R1 | SOTA mAP | Reproduced best epoch | Reproduced R1 | Reproduced mAP | Delta R1 | Delta mAP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| CUHK-PEDES | 9 | 70.923 | 63.503 | 9 | 70.809 | 63.469 | -0.114 | -0.034 |
| ICFG-PEDES | 9 | 62.152 | 36.110 | 9 | 62.006 | 35.923 | -0.146 | -0.187 |
| RSTPReid | 49 | 61.850 | 46.304 | 49 | 61.950 | 46.274 | +0.100 | -0.030 |
| PAB | 49 | 76.138 | 86.145 | 49 | 75.784 | 86.010 | -0.354 | -0.135 |

