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
- **Built-in Chat Interface** - Terminal-based chat for inference with compressed models

---

## Preliminary Results

> **Disclaimer**: The following results are from internal testing on limited evaluation subsets. They have NOT been independently reproduced or validated against standardized benchmark protocols. These numbers should be treated as indicative, not definitive.

### Perplexity (WikiText-2 Test Subset)

Evaluated on WikiText-2 test split (256 context chunks of 1024 tokens).

| Model | Params | FP16 Size | Bits | Compressed | Ratio | FP16 PPL | TMG-Q PPL | Delta |
|-------|--------|-----------|------|------------|-------|----------|-----------|-------|
| GPT-2 Medium | 350M | 709 MB | 3-bit | 133 MB | 5.3x | 33.40 | 34.29 | +2.6% |
| GPT-2 Large | 774M | 1548 MB | 2-bit | 193 MB | 8.0x | 29.43 | 29.66 | +0.8% |
| TinyLlama 1.1B | 1.1B | 2200 MB | 4-bit | ~600 MB | 3.7x | -- | Coherent | -- |

The TinyLlama 4-bit export has been verified to produce coherent English text in interactive chat, but formal PPL measurement with the full Hessian+SVD pipeline on LLaMA architectures has not yet been completed.

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
| **TMG-Q** | **4-bit** | **TinyLlama-1.1B** | **TBD** | **Verified coherent generation only** |
| **TMG-Q** | **3-bit** | **GPT-2 Medium** | **34.29** | **Preliminary, internal eval** |

> **Honest assessment**: TMG-Q has not yet been evaluated on the same models (LLaMA-7B/13B) or with the same standardized protocols used by GPTQ, QuIP#, AQLM, and AWQ. A direct apples-to-apples comparison is required before any claims of competitiveness can be made. This is the top priority in our roadmap.

---

## Quick Start

### Requirements

```bash
pip install torch transformers numpy
```

### Compress and Export a HuggingFace Model

Compress any HuggingFace model into a single shareable `.pt` file:

```bash
python tmgq_export_llama.py
```

Output: `TinyLlama_4bit_TMGQ.pt` - a self-contained compressed model file.

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

1. **Limited benchmark coverage**: Results are only available for GPT-2 Medium/Large and a custom 85M nanoGPT model. No evaluation has been performed on LLaMA-7B/13B/70B or other standard benchmark models.

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
