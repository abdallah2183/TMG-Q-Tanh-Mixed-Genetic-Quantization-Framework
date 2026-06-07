# TMG-Q Experiment Report

Date: 2026-06-05

## Environment

- GPU: NVIDIA GeForce RTX 5060 Ti, 16 GB VRAM
- Python environment: local CUDA-enabled Python with PyTorch and Transformers
- Torch: 2.12.0.dev20260408+cu128
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
- Evaluation: WikiText-2, sequence length 128

The local Python environment had a Torch/Torchvision CUDA mismatch. TMG-Q now disables optional torchvision imports inside its text-only scripts through `tmgq_transformers_compat.py`.

## Key Engineering Changes

1. True 2-bit, 3-bit, and 4-bit INT32 packing through `pack_nbit` / `unpack_nbit`.
2. Calibration-aware export using WikiText-2 activation statistics as Hessian-diagonal proxies.
3. Hessian-guided rounding between floor, round, and ceil.
4. Row-wise clipping search to reduce outlier damage.
5. Sparse outlier residuals for sub-4-bit recovery.
6. Optional low-rank SVD residuals for structural error recovery.
7. Learned Lloyd-style group codebooks for non-uniform 2-bit and 3-bit quantization.
8. Per-layer mixed-precision policies.
9. Automated sweep runner, PPL evaluator, checkpoint size report, and layer-level benchmarks.

## TinyLlama Quick Sweep

These results used 8 WikiText-2 samples for fast search.

| Config | Calibration | PPL | Delta vs Baseline | Matched Linear Ratio |
|---|---:|---:|---:|---:|
| Baseline FP/BF16 | no | 13.5373 | 0.0000 | 1.00x |
| 4-bit, g128 | no | 16.8124 | +3.2751 | 3.76x |
| 4-bit, g64 | no | 15.4834 | +1.9461 | 3.56x |
| 4-bit, g64 | yes | 14.8477 | +1.3104 | 3.56x |
| 4-bit, g32 | yes | 14.9026 | +1.3653 | 3.20x |
| 3-bit, g128 | no | 1142.3937 | +1128.8564 | 4.64x |
| 3-bit, g128, 0.1% residual | no | 94.7382 | +81.2009 | 4.53x |
| 3-bit, g128, 1% residual | no | 71.5028 | +57.9655 | 3.76x |
| 3-bit, g128, 1% residual | yes | 18.1479 | +4.6106 | 3.76x |
| 3-bit, g64, 1% residual | yes | 17.5485 | +4.0112 | 3.56x |
| 3-bit, g32, 1% residual | yes | 16.3927 | +2.8554 | 3.20x |
| 3-bit, g32, 1% residual, SVD rank 8 | yes | 16.4422 | +2.9049 | 3.13x |
| 3-bit codebook, g128 | yes | 17.4571 | +3.9198 | 3.81x |
| 3-bit codebook, g64 | yes | 16.1331 | +2.5958 | 3.08x |
| 3-bit codebook, g32 | yes | 14.5112 | +0.9739 | 2.22x |
| 3-bit codebook, g32, 0.1% residual | yes | 14.4693 | +0.9320 | 2.20x |
| 2-bit, g128, 1% residual | yes | 617.9627 | +604.4254 | 5.25x |
| 2-bit, g128, 1% residual, SVD rank 8 | yes | 431.8017 | +418.2644 | 5.07x |
| 2-bit codebook, g128 | yes | 43.1953 | +29.6580 | 6.40x |
| 2-bit codebook, g64 | yes | 33.1659 | +19.6286 | 5.33x |
| 2-bit codebook, g32, 0.1% residual | yes | 25.9620 | +12.4247 | 3.92x |
| Mixed balanced, g128 | yes | 17.1270 | +3.5897 | 3.80x |
| Mixed aggressive, g64, 0.1% residual | yes | 22.3075 | +8.7702 | 4.14x |
| Adaptive mixed 2/3/4-bit, g64, 0.1% residual | yes | 14.5741 | +1.0368 | 3.51x |

## 32-Sample Verification

These results reevaluated the best checkpoints without re-exporting.

| Config | PPL | Delta vs Baseline |
|---|---:|---:|
| Baseline FP/BF16 | 17.7996 | 0.0000 |
| 4-bit, g64, calibrated | 19.0321 | +1.2325 |
| 3-bit, g32, 1% residual, calibrated | 20.1055 | +2.3059 |
| 3-bit, g64, 1% residual, calibrated | 21.2846 | +3.4850 |
| 3-bit codebook, g32, calibrated | 19.1697 | +1.3701 |
| 3-bit codebook, g32, 0.1% residual, calibrated | 19.0768 | +1.2772 |
| 3-bit codebook, g64, calibrated | 20.6939 | +2.8943 |
| 2-bit codebook, g32, 0.1% residual, calibrated | 32.8910 | +15.0914 |
| Adaptive mixed 2/3/4-bit, g64, 0.1% residual, calibrated | 18.9206 | +1.1210 |

## Interpretation

The strongest production-like setting is currently:

```text
Adaptive 2/3/4-bit, group_size=64, 0.1% sparse residual, calibrated
PPL: 18.9206 vs 17.7996 baseline on 32 samples
Matched linear compression: 3.51x
Layer distribution: 12 at 2-bit, 31 at 3-bit, 111 at 4-bit
```

The strongest research setting is currently:

```text
3-bit codebook, group_size=32, 0.1% sparse residual, calibrated
PPL: 19.0768 vs 17.7996 baseline on 32 samples
Matched linear compression: 2.20x
```

The most important finding is that 3-bit raw quantization collapses, but sparse residual plus calibration recovers it dramatically:

```text
3-bit raw, g128: PPL 1142.3937
3-bit + 1% residual, calibrated, g128: PPL 18.1479
```

2-bit remains unstable even with 1% residual and calibration. It likely needs a rotation/preconditioning method, additive codebooks, or a stronger low-rank residual path.

Low-rank SVD residuals improved synthetic layer tests and reduced 2-bit TinyLlama damage from PPL 617.9627 to 431.8017, but this is still unusable. For the best 3-bit TinyLlama configuration, SVD rank 8 slightly worsened PPL (16.4422 vs 16.3927) while lowering compression ratio (3.13x vs 3.20x), so the current best 3-bit setting remains residual-only.

Learned codebooks were substantially stronger than linear quantization below 4-bit. On the synthetic outlier benchmark, 2-bit cosine similarity improved from 0.7754 to 0.9369. On TinyLlama, calibrated 2-bit codebooks reduced quick-sweep PPL from hundreds or thousands to 25.962 at group size 32 with a 0.1% residual. This is not yet competitive with 3-bit or 4-bit, but it is the first usable direction found for the 2-bit path.

Mixed precision produced a useful high-compression point (3.80x at PPL 17.127 in the quick sweep), but did not beat the best uniform 4-bit setting on quality. The balanced and aggressive policies remain available for future per-layer sensitivity search.

The adaptive policy supersedes the fixed mixed policies for the quality-first operating point. It probes representative rows using Hessian-weighted NMSE, selects among 2-bit codebook, 3-bit codebook, and 4-bit linear quantization, then quantizes the full layer only once. On the 32-sample verification it improved PPL from 19.0321 for uniform 4-bit to 18.9206 while retaining a similar 3.51x matched compression ratio.

## GPT-2 Base Modern-Packer Verification

The modern CUDA exporter now supports GPT-2 `Conv1D` projections and token embeddings. These are fresh 8-sample, length-128 WikiText-2 measurements and are separate from the older preliminary README claims.

| Configuration | PPL | Checkpoint | Full ratio |
|---|---:|---:|---:|
| BF16 baseline | 62.7657 | ~237.4 MiB | 1.00x |
| Conv1D 4-bit, g64, calibrated; vocabulary BF16 | 67.3786 | 121.2 MiB | 1.96x |
| Conv1D + input embedding 4-bit; lm_head BF16 | 67.1946 | 141.9 MiB | 1.67x |
| Fully tied vocabulary 4-bit | 1744.5986 | 68.2 MiB | 3.48x |
| Fully tied vocabulary 4-bit + rank-64 residual | 1174.9785 | 74.5 MiB | 3.19x |
| Fully tied vocabulary 4-bit + rank-128 logit distillation | 58.8744 | 80.7 MiB | 2.94x |
| Fully tied vocabulary 4-bit + rank-64 logit distillation | 58.7533 | 74.5 MiB | 3.19x |
| Adaptive 3/4-bit + rank-64 distillation | 60.4898 | 70.8 MiB | 3.35x |
| Adaptive 3/4-bit + rank-32 distillation | 61.2229 | 67.7 MiB | 3.51x |
| Fixed balanced 3/4-bit + rank-64 distillation | 66.1131 | 70.5 MiB | 3.37x |

The quality-first GPT-2 checkpoint is the uniform 4-bit model with a rank-64 distilled vocabulary. It scores 58.7533 PPL versus 58.5241 for BF16, a 0.39% increase, at 3.19x full-file compression.

The compression-first checkpoint that still passes the 5% quality gate uses two sensitivity-selected 3-bit layers, 47 4-bit packed matrices, and a rank-32 distilled vocabulary. It scores 61.2229 PPL, a 4.61% increase, at 3.51x full-file compression. The fixed balanced policy was rejected because its PPL increased by 12.97%, demonstrating that per-layer sensitivity selection is necessary.

Further experiments did not replace this operating point:

| Experiment | PPL | Full ratio | Verdict |
|---|---:|---:|---|
| Adaptive 6-layer 3-bit + codebook QAT | 62.3321 | 3.51x | Rejected: +6.51% PPL |
| Rank-24 distilled vocabulary | 63.1513 | 3.55x | Rejected: +7.91% PPL |
| FP8 vocabulary residual | 61.4633 | 3.59x | Rejected: just above 5% gate |
| Row-scaled INT8 vocabulary residual | 61.6137 | 3.58x | Rejected: +5.28% PPL |
| FP8 linear scales/zeros | 4622.9760 | 3.60x | Rejected: catastrophic scale error |

QAT-Lite support was added for training packed codebooks without changing their storage size. It reduced calibration KL but did not improve held-out WikiText-2 PPL in the tested configurations, so the feature remains experimental rather than part of the recommended checkpoint.

## Repro Commands

```powershell
python tmgq_sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tag tinyllama_real_quick --max-length 128 --samples 8 --calib-samples 8 --calib-length 128 --configs '4:64:0:true,3:32:0.01:true'
```

```powershell
python tmgq_eval_ppl.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --checkpoint sweep_results/tinyllama_real_quick/tinyllama_real_quick_4bit_g64_o0_cal.pt --max-length 128 --samples 32
```

```powershell
python tmgq_sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tag tinyllama_adaptive_cuda --skip-baseline --configs '3:64:0.001:true:0:linear:adaptive' --calib-samples 8 --calib-length 128 --samples 8 --max-length 128 --adaptive-2bit-nmse 0.04 --adaptive-3bit-nmse 0.015 --adaptive-probe-rows 64 --quant-device cuda
```
