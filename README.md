# TMG-Q: Tanh-Mixed Genetic Quantization Framework

**A Post-Training Quantization Framework for Compressing Large Language Models**

*Exploring extreme compression (4-bit / 3-bit / 2-bit) via Hessian-guided Error Diffusion and Physical INT32 Bit-Packing*

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white)
![Status](https://img.shields.io/badge/Status-Experimental-orange)

---

## Overview

**TMG-Q** is an experimental post-training quantization (PTQ) framework that compresses large language models down to **4-bit, 3-bit, and 2-bit** precision. It combines **Hessian-guided Error Diffusion**, **SVD Residual Recovery**, and **Physical INT32 Bit-Packing** to achieve aggressive compression ratios without requiring retraining.

> **Note**: This is an active research project. Results are preliminary and have been evaluated on limited internal benchmarks. Full standardized evaluation against established methods is planned but not yet completed. See [Limitations](#limitations) for details.

### Key Features

- **Physical INT32 Bit-Packing** - Packs 8x (4-bit) or 10x (3-bit) weights into a single INT32 container, achieving real physical memory reduction
- **Hessian-Guided Error Diffusion** - Uses second-order sensitivity information to redistribute quantization error across less-critical weights
- **SVD Residual Recovery** - Recovers the most damaging structural deviations via truncated Singular Value Decomposition
- **Export and Share** - Compress any HuggingFace model into a single portable `.pt` file
- **GPT-2 Conv1D Support** - Uses the modern packed path for native GPT-2 projection layers
- **Quantized Vocabulary Experiments** - Supports tied token embeddings, quantized lm_head, and low-rank vocabulary residuals
- **Vocabulary Distillation** - Trains only a compact low-rank correction against the original teacher logits
- **QAT-Lite Codebooks** - Optionally tunes selected packed 2/3-bit codebooks without changing their index payload
- **Metadata Repacking Lab** - Tests FP8 or row-scaled INT8 storage for residual and quantization metadata
- **Built-in Chat Interface** - Terminal-based chat for inference with compressed models

---

## Verified Results

> **Disclaimer**: These are internal WikiText-2 measurements on limited subsets. They have not been independently reproduced and should not be compared directly with papers using different models, context lengths, or evaluation protocols.

### Perplexity (WikiText-2 Test Subset)

Evaluated on 32 chunks of 128 tokens. GPT-2 ratios cover the full checkpoint file. TinyLlama ratios compare the matched quantized matrix payload with its FP16 equivalent.

| Model and operating point | Baseline PPL | TMG-Q PPL | PPL increase | Compression |
|---|---:|---:|---:|---:|
| GPT-2 Base, quality: 4-bit + distilled rank 64 | 58.5241 | 58.7533 | 0.39% | 3.19x full file |
| GPT-2 Base, compression: adaptive 3/4-bit + distilled rank 32 | 58.5241 | 61.2229 | 4.61% | 3.51x full file |
| TinyLlama 1.1B, adaptive 2/3/4-bit + 0.1% residual | 17.7996 | 18.9206 | 6.30% | 3.51x matched matrices |

The detailed experiment history, rejected configurations, and reproduction commands are in [TMGQ_EXPERIMENT_REPORT.md](TMGQ_EXPERIMENT_REPORT.md).

Older GPT-2 Medium/Large and NanoGPT numbers remain historical project observations but have not yet been reproduced with the modern packed exporter and current evaluation harness.

### Functional Logic (Pass@1) - NanoGPT Only

Tested on a custom 85M parameter coding-focused nanoGPT model. Pass@1 evaluated by executing generated Python via AST parsing.

| Precision | Model Size | Pass@1 | Notes |
|-----------|------------|--------|-------|
| FP16 (Baseline) | 172.0 MB | 72.0% | Original model |
| 4-bit (TMG-Q) | 44.1 MB | 72.0% | Full Hessian + SVD pipeline |
| 3-bit (TMG-Q) | 33.5 MB | 72.0% | Full Hessian + SVD pipeline |
| 2-bit (TMG-Q) | 22.9 MB | 72.0% | Full Hessian + SVD pipeline |
| 2-bit (Naive RTN) | 22.9 MB | 0.0% | Round-to-nearest only (no diffusion) |

> **Important caveat on 2-bit results**: The 72% Pass@1 retention at 2-bit was observed on a small 85M parameter model with a narrow task domain (simple Python functions). This does NOT imply that 2-bit TMG-Q will retain equivalent quality on larger, general-purpose LLMs. Sub-4-bit quantization on billion-parameter models without calibration data typically causes significant degradation, as demonstrated by our own TinyLlama experiments where 3-bit RTN (without Hessian/SVD) resulted in complete output collapse. Proper calibration-aware quantization for large models at 2-3 bit is an active area of future work.

### Comparison with Established Methods

The following table shows **published results from other methods** for reference. These are NOT head-to-head comparisons on the same hardware or evaluation setup.

| Method | Bits | Model | WikiText-2 PPL | Notes |
|--------|------|-------|----------------|-------|
| GPTQ | 4-bit | LLaMA-7B | ~6.1 | Calibration-based, widely adopted |
| GPTQ | 3-bit | LLaMA-7B | ~8.1 | With grouping |
| QuIP# | 2-bit | LLaMA-7B | ~8.5 | Incoherence processing + lattice codebooks |
| AQLM | 2-bit | LLaMA-7B | ~7.9 | Additive codebook quantization |
| AWQ | 4-bit | LLaMA-7B | ~5.8 | Activation-aware weight quantization |
| **TMG-Q** | **adaptive 3/4-bit** | **GPT-2 Base** | **61.22** | **Internal 32 x 128-token eval; 3.51x full-file ratio** |
| **TMG-Q** | **adaptive 2/3/4-bit** | **TinyLlama-1.1B** | **18.92** | **Internal 32 x 128-token eval** |

> **Honest assessment**: TMG-Q now has reproducible internal GPT-2 Base and TinyLlama measurements, but it has not been evaluated on the same models or standardized protocols used by GPTQ, QuIP#, AQLM, and AWQ. Direct comparisons are still required before claiming competitiveness.

---

## Quick Start

### Requirements

```bash
pip install torch transformers datasets numpy
```

### Compress and Export a HuggingFace Model

Compress any HuggingFace model into a single shareable `.pt` file:

```bash
python tmgq_export_llama.py
```

Output: `TinyLlama_4bit_TMGQ.pt` - a self-contained compressed model file.

For a stronger calibrated export, use WikiText-2 activation statistics for Hessian-guided rounding:

```bash
python tmgq_export_llama.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --bits 4 \
  --calibrate \
  --calib-samples 64 \
  --calib-length 256 \
  --output TinyLlama_4bit_TMGQ_calibrated.pt
```

The exporter supports true physical `2-bit`, `3-bit`, and `4-bit` packing:

```bash
python tmgq_export_llama.py --bits 3 --output TinyLlama_3bit_TMGQ.pt
```

Use learned group codebooks for stronger sub-4-bit quality:

```bash
python tmgq_export_llama.py \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --bits 3 \
  --group-size 32 \
  --quantizer codebook \
  --outlier-fraction 0.001 \
  --calibrate \
  --output TinyLlama_3bit_Codebook_TMGQ.pt
```

Use per-layer mixed precision:

```bash
# Boundary layers at 4-bit linear, middle layers at 3-bit codebook
python tmgq_export_llama.py --mixed-policy balanced --group-size 128 --calibrate

# Attention at 3-bit codebook, middle MLP layers at 2-bit codebook
python tmgq_export_llama.py --mixed-policy aggressive --group-size 64 --calibrate
```

GPU adaptive export:

```powershell
python tmgq_export_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mixed-policy adaptive --group-size 64 --outlier-fraction 0.001 --calibrate --adaptive-probe-rows 64 --quant-device cuda
```

Global payload-budget optimization profiles 2/3/4-bit candidates and rejects mathematically impossible targets:

```powershell
python tmgq_export_llama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --mixed-policy budget --target-ratio 4.0 --group-size 64 --calibrate --quant-device cuda
```

Sweeps can enforce a quality gate. The following marks checkpoints as rejected when PPL increases by more than 5%:

```powershell
python tmgq_sweep.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --configs "3:64:0.001:true:0:linear:budget" --target-ratio 4.0 --max-ppl-increase 5 --quant-device cuda
```

Distill a tied quantized GPT-2 vocabulary while keeping all packed transformer weights frozen:

```powershell
python tmgq_distill_vocab.py --model gpt2 --checkpoint sweep_results/gpt2_full_4bit_g64_cal.pt --output sweep_results/gpt2_vocab4_distilled_r128.pt --rank 128 --steps 250 --samples 64 --sequence-length 128 --initialize-svd
```

Experimental QAT-Lite for sub-4-bit codebooks:

```powershell
python tmgq_distill_vocab.py --model gpt2 --checkpoint packed.pt --output qat.pt --rank 32 --tune-quant-layers --tune-max-bits 3
```

Verified GPT-2 Base operating points on 32 WikiText-2 chunks:

| Operating point | PPL | PPL increase | Full compression |
|---|---:|---:|---:|
| BF16 baseline | 58.5241 | -- | 1.00x |
| Quality: 4-bit + distilled rank 64 | 58.7533 | 0.39% | 3.19x |
| Compression: adaptive 3/4-bit + distilled rank 32 | 61.2229 | 4.61% | 3.51x |

Customize the target model and precision by editing the function call at the bottom of the script:

```python
export_huggingface_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    bits=4,
    export_path="TinyLlama_4bit_TMGQ.pt"
)
```

### Chat with a Compressed Model

Run the terminal chat interface with an exported `.pt` file:

```bash
python tmgq_friend_chat.py
```

```
==================================================
TMG-Q Ultra Native Terminal Chat
Type 'exit' or 'quit' to end the session.
==================================================

You: What is machine learning?
Bot: Machine learning is a subset of artificial intelligence that involves
     training algorithms to learn patterns from data...
```

### Evaluate Perplexity (GPT-2 Family)

```bash
# GPT-2 Medium at 3-bit with full Hessian + SVD pipeline
python TMG-Q/scripts/tmgq_ultra_hf.py --model gpt2-medium --bits 3 --test

# GPT-2 Large at 2-bit
python TMG-Q/scripts/tmgq_ultra_hf.py --model gpt2-large --bits 2 --test
```

Evaluate a packed exported model against WikiText-2:

```bash
python tmgq_eval_ppl.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --checkpoint TinyLlama_4bit_TMGQ_calibrated.pt
```

### Compress Custom NanoGPT Models

```bash
python omegaquant_nanogpt.py
python nano_chat_ui.py
```

---

## Repository Structure

```
TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework/
|
|-- tmgq_packer.py              # Core INT32 bit-packing engine (3-bit and 4-bit)
|-- tmgq_export_llama.py        # Compress HuggingFace models to portable .pt file
|-- tmgq_friend_chat.py         # Load compressed .pt and run terminal chat
|
|-- omegaquant_nanogpt.py       # NanoGPT-specific quantization (Hessian + SVD)
|-- nano_chat_ui.py             # Gradio chat UI for compressed NanoGPT models
|
|-- TMG-Q/                      # Extended framework
|   |-- core/                   # Core quantization algorithms
|   |-- scripts/                # CLI evaluation tools
|   |   |-- tmgq_ultra_hf.py   # HuggingFace benchmark CLI
|   |-- evolution/              # Genetic algorithm components
|   |-- utils/                  # Utility functions
|   |-- tests/                  # Unit tests
|   |-- docs/                   # Documentation
|
|-- setup.py
|-- LICENSE
|-- README.md
```

---

## Methodology

### 1. Hessian Diagonal Extraction

Calibrate the model using forward passes on sampled texts to approximate the diagonal of the Hessian matrix. This yields a per-weight sensitivity profile that guides quantization precision allocation.

### 2. Error Diffusion Waves

The residual quantization error (E = W_orig - W_quant) is re-dispersed across the remaining weights in the same layer tensor, scaled by the measured Hessian trace. This prevents error accumulation in sensitive regions.

### 3. Truncated SVD Recovery

The remaining un-diffusible error is factored using Singular Value Decomposition. The top singular values are restored to neutralize the most damaging structural deviations.

### 4. Physical INT32 Packing

Quantized integer weights are physically packed into INT32 containers using bitwise operations:
- **4-bit**: 8 weights per INT32 (4 x 8 = 32 bits)
- **3-bit**: 10 weights per INT32 (3 x 10 = 30 bits)

At inference time, weights are unpacked on-the-fly and dequantized for matrix multiplication.

---

## Limitations

This project is in an early experimental stage. Known limitations include:

1. **Limited benchmark coverage**: The modern exporter has verified internal measurements for GPT-2 Base and TinyLlama 1.1B, plus historical observations for GPT-2 Medium/Large and a custom 85M nanoGPT model. No evaluation has been performed on LLaMA-7B/13B/70B.

2. **No standardized evaluation protocol**: PPL numbers were measured on internal evaluation subsets, not using the exact protocols from GPTQ/QuIP#/AQLM papers. Direct numerical comparison is therefore not valid.

3. **Missing standard benchmarks**: No evaluation on HumanEval, MBPP, C4 perplexity, MMLU, or other widely-used LLM benchmarks.

4. **2-bit results need validation**: The 2-bit results on nanoGPT (72% Pass@1) were achieved on a small model with narrow task scope. Generalization to larger models is unproven.

5. **No calibration for LLM export**: The current HuggingFace export pipeline (`tmgq_export_llama.py`) uses basic RTN quantization without the Hessian/SVD pipeline. Integrating the full pipeline for billion-parameter models is planned.

6. **CPU-only dequantization**: The current QuantizedLinear layer dequantizes weights in Python loops. A CUDA kernel implementation would significantly improve inference speed.

---

## Roadmap

- [x] Physical INT32 3-bit and 4-bit packing engine
- [x] HuggingFace model export pipeline (basic RTN)
- [x] Terminal chat inference from compressed .pt files
- [x] BFloat16 runtime for LLaMA-family activation stability
- [ ] Integrate Hessian + SVD pipeline into HuggingFace exporter
- [ ] Standardized WikiText-2 PPL on LLaMA-7B (direct GPTQ/AWQ comparison)
- [ ] C4 perplexity evaluation
- [ ] HumanEval / MBPP code generation benchmarks
- [ ] CUDA kernel for packed weight dequantization
- [ ] GGUF export format compatibility
- [ ] Streaming token generation

---

## License

Copyright 2026 Abdullah Salem Saleh Al-Faqeer. All Rights Reserved.

---

## Citation

```
@software{tmgq2026,
  author = {Al-Faqeer, Abdullah Salem Saleh},
  title  = {TMG-Q: Tanh-Mixed Genetic Quantization Framework},
  year   = {2026},
  url    = {https://github.com/abdallah2183/TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework}
}
```
