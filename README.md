<div align="center">

# TMG-Q

### Tanh-Mixed Genetic Quantization Framework

**Real 2/3/4-bit weight packing, calibration-aware mixed precision, and measured LLM compression.**

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-verified-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-2EA44F)](LICENSE)
[![Research](https://img.shields.io/badge/status-experimental-F59E0B)](#research-status)

[Results](#verified-results) | [How it works](#how-it-works) | [Quick start](#quick-start) | [Reproduce](#reproduce-the-results) | [Report](TMGQ_EXPERIMENT_REPORT.md)

</div>

---

## Research Paper

The five-page technical paper includes the method, equations, experimental protocol, verified results, negative findings, figures, and academic references:

**[Download the TMG-Q Research Paper (PDF)](paper/TMG-Q_Research_Paper_2026.pdf)**

## What TMG-Q Is

TMG-Q is an experimental post-training quantization framework for Hugging Face causal language models. It stores quantized weights in physically packed `INT32` tensors and selects 2-bit, 3-bit, or 4-bit precision using calibration statistics and layer sensitivity.

The current implementation includes:

- True physical 2-bit, 3-bit, and 4-bit packing.
- CUDA quantization with CPU fallback.
- GPT-2 `Conv1D`, linear layer, embedding, and tied `lm_head` support.
- Linear and learned group-codebook quantizers.
- Hessian-proxy calibration and sensitivity-guided rounding.
- Adaptive and global-budget mixed-precision policies.
- Sparse outlier and optional low-rank residual recovery.
- Vocabulary logit distillation for compressed tied embeddings.
- Perplexity evaluation, checkpoint accounting, sweeps, and quality gates.

> TMG-Q currently optimizes checkpoint size and reconstruction quality. It does not yet provide a fused packed CUDA inference kernel, so compression does not automatically mean faster token generation.

## Verified Results

All values below were reproduced locally on **June 7, 2026** using an **NVIDIA GeForce RTX 5060 Ti 16 GB**. Perplexity was measured on the WikiText-2 test split using **32 non-overlapping chunks of 128 tokens** in BF16 runtime.

### GPT-2 Base

| Operating point | Precision strategy | Checkpoint | Size reduction | WikiText-2 PPL | PPL change |
|---|---|---:|---:|---:|---:|
| BF16 baseline | Uncompressed | 237.4 MiB | 0.0% | 58.5241 | - |
| **Quality** | 4-bit weights + rank-64 distilled vocabulary | **74.5 MiB (3.19x)** | **68.7%** | **58.7533** | **+0.39%** |
| **Compression** | Adaptive 3/4-bit + rank-32 distilled vocabulary | **67.7 MiB (3.51x)** | **71.5%** | **61.2229** | **+4.61%** |

The compression operating point contains **2 layers at 3-bit and 47 layers at 4-bit**. It is the smallest tested GPT-2 checkpoint that remains inside the project's 5% perplexity quality gate.

### TinyLlama 1.1B

| Metric | BF16 baseline | Adaptive TMG-Q |
|---|---:|---:|
| WikiText-2 PPL | 17.7996 | **18.9206** |
| PPL change | - | **+6.30%** |
| Full checkpoint payload | 2,098.2 MiB | **778.0 MiB** |
| Full checkpoint compression | 1.00x | **2.70x (63.0% smaller)** |
| Matched quantized matrices | 1,848.0 MiB | **526.9 MiB** |
| Matched matrix compression | 1.00x | **3.51x (71.5% smaller)** |
| Layer allocation | - | **12 x 2-bit, 31 x 3-bit, 111 x 4-bit** |

TinyLlama uses adaptive 2/3/4-bit quantization, group size 64, calibration, and a 0.1% sparse residual. The full-file ratio is lower than the matrix ratio because the checkpoint also contains uncompressed parameters and metadata.

### What These Numbers Mean

- **Checkpoint ratio** compares the complete serialized TMG-Q file with the model's unique FP16 parameter payload.
- **Matched matrix ratio** compares only tensors replaced by packed quantized modules.
- **PPL change** is relative, calculated as `(quantized PPL / baseline PPL - 1) x 100`.
- Results from different models are not directly comparable because their baseline perplexities differ.

These are internal, reproducible measurements, not an independent benchmark or a head-to-head result against GPTQ, AWQ, AQLM, or QuIP#. The raw experiment history and rejected configurations are documented in [TMGQ_EXPERIMENT_REPORT.md](TMGQ_EXPERIMENT_REPORT.md).

## How It Works

```text
Hugging Face model
        |
        v
WikiText-2 calibration -> activation/Hessian proxies
        |
        v
Layer sensitivity probing
        |
        +---- 2-bit learned codebook
        +---- 3-bit learned codebook
        +---- 4-bit linear quantization
        |
        v
Optional sparse or low-rank residual recovery
        |
        v
Physical INT32 packing -> portable .pt checkpoint
        |
        v
PPL evaluation + real byte accounting + quality gate
```

### Packing

| Precision | Values stored per INT32 | Payload use |
|---|---:|---:|
| 2-bit | 16 | 32 / 32 bits |
| 3-bit | 10 | 30 / 32 bits |
| 4-bit | 8 | 32 / 32 bits |

The reported size includes scales, zero points or codebooks, sparse residuals, low-rank factors, and serialized checkpoint overhead. It is not a theoretical `parameter_count x bits` estimate.

### Adaptive Precision

For each projection, TMG-Q probes representative rows and measures Hessian-weighted normalized reconstruction error. It accepts the lowest precision that satisfies the configured error threshold, then quantizes the complete layer once.

The global-budget mode instead profiles 2/3/4-bit candidates and solves a checkpoint payload allocation problem under a requested compression target.

## Quick Start

### Install

```bash
git clone https://github.com/abdallah2183/TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework.git
cd TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework
pip install torch transformers datasets numpy
```

For GPU quantization, install a CUDA-enabled PyTorch build appropriate for your system.

### Export A Calibrated 4-Bit Model

```bash
python tmgq_export_llama.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --bits 4 \
  --group-size 64 \
  --calibrate \
  --quant-device cuda \
  --output TinyLlama_TMGQ_4bit.pt
```

### Export Adaptive 2/3/4-Bit

```bash
python tmgq_export_llama.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --mixed-policy adaptive \
  --group-size 64 \
  --outlier-fraction 0.001 \
  --adaptive-2bit-nmse 0.04 \
  --adaptive-3bit-nmse 0.015 \
  --adaptive-probe-rows 64 \
  --calibrate \
  --quant-device cuda \
  --output TinyLlama_TMGQ_adaptive.pt
```

### Evaluate And Measure

```bash
python tmgq_eval_ppl.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --checkpoint TinyLlama_TMGQ_adaptive.pt \
  --max-length 128 \
  --samples 32

python tmgq_checkpoint_report.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  TinyLlama_TMGQ_adaptive.pt
```

### Run The Compressed Model

```bash
python tmgq_friend_chat.py
```

## Reproduce The Results

### TinyLlama Adaptive Sweep

```bash
python tmgq_sweep.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --tag tinyllama_adaptive_cuda \
  --skip-baseline \
  --configs "3:64:0.001:true:0:linear:adaptive" \
  --calib-samples 8 \
  --calib-length 128 \
  --samples 8 \
  --max-length 128 \
  --adaptive-2bit-nmse 0.04 \
  --adaptive-3bit-nmse 0.015 \
  --adaptive-probe-rows 64 \
  --quant-device cuda
```

Re-evaluate the selected checkpoint on 32 chunks:

```bash
python tmgq_eval_ppl.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --checkpoint sweep_results/tinyllama_adaptive_cuda/tinyllama_adaptive_cuda_3bit_g64_o0.001_svd0_linear_adaptive_cal.pt \
  --max-length 128 \
  --samples 32
```

### GPT-2 Vocabulary Distillation

```bash
python tmgq_distill_vocab.py \
  --model gpt2 \
  --checkpoint sweep_results/gpt2_full_4bit_g64_cal.pt \
  --output sweep_results/gpt2_vocab4_distilled_r64.pt \
  --rank 64 \
  --steps 250 \
  --samples 64 \
  --sequence-length 128 \
  --initialize-svd
```

## Main Tools

| File | Purpose |
|---|---|
| `tmgq_packer.py` | Packed quantized modules and 2/3/4-bit codecs |
| `tmgq_export_llama.py` | Hugging Face calibration, quantization, and export |
| `tmgq_budget_optimizer.py` | Global mixed-precision payload optimizer |
| `tmgq_distill_vocab.py` | Low-rank vocabulary/logit distillation |
| `tmgq_eval_ppl.py` | WikiText-2 perplexity evaluation |
| `tmgq_checkpoint_report.py` | Full-file and matched-payload byte accounting |
| `tmgq_sweep.py` | Automated export/evaluation sweeps and quality gates |
| `tmgq_friend_chat.py` | Packed checkpoint loading and terminal inference |
| `test_tmgq_packer.py` | Packing and quantized-module tests |
| `test_tmgq_budget_optimizer.py` | Budget optimizer tests |

## Research Status

TMG-Q is an active research prototype.

**Verified**

- Physical 2/3/4-bit packing and exact round-trip decoding.
- GPT-2 Base and TinyLlama 1.1B export, reload, and WikiText-2 evaluation.
- CUDA quantization on an RTX 5060 Ti 16 GB.
- Real checkpoint and tensor-payload accounting.
- Adaptive mixed precision and vocabulary distillation.

**Not yet established**

- Standard full-length WikiText-2 protocol used by published baselines.
- MMLU, C4, HumanEval, MBPP, GSM8K, or TruthfulQA results.
- Direct same-model comparison with GPTQ, AWQ, AQLM, or QuIP#.
- Production-speed packed CUDA inference kernels.
- Stable quality-preserving 2-bit operation across entire billion-parameter models.
- Validation on 7B, 13B, or larger models.

No claim of state-of-the-art quality or 30x lossless compression is made. The strongest verified full-checkpoint result is currently **3.51x on GPT-2 Base within a 5% PPL gate**.

## Roadmap

- [x] Physical INT32 packing for 2-bit, 3-bit, and 4-bit weights
- [x] Calibration-aware CUDA exporter
- [x] Learned codebooks and sparse residual recovery
- [x] Adaptive and budget-constrained mixed precision
- [x] GPT-2 embeddings, tied head, and vocabulary distillation
- [x] Automated PPL, size reporting, sweeps, and tests
- [ ] Fused CUDA kernels for packed matrix multiplication
- [ ] Standardized lm-evaluation-harness integration
- [ ] Same-model GPTQ/AWQ/AQLM comparison
- [ ] Rotation or incoherence processing for robust 2-bit quantization
- [ ] Safetensors and GGUF export
- [ ] Evaluation on 7B-class models

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@software{tmgq2026,
  author = {Al-Faqeer, Abdullah Salem Saleh},
  title = {TMG-Q: Tanh-Mixed Genetic Quantization Framework},
  year = {2026},
  url = {https://github.com/abdallah2183/TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework}
}
```
