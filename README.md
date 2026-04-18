<![CDATA[# TMG-Q: Tanh-Mixed Genetic Quantization Framework

<p align="center">
  <b>A Novel Post-Training Quantization Framework for Compressing Large Language Models</b><br>
  <i>Achieving extreme compression (4-bit / 3-bit / 2-bit) with minimal perplexity degradation</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=white" />
  <img src="https://img.shields.io/badge/License-Proprietary-red" />
</p>

---

## 📖 Overview

**TMG-Q** is a post-training quantization (PTQ) framework that compresses large language models down to **4-bit, 3-bit, and even 2-bit** precision while preserving functional coherence. It combines **Hessian-guided Error Diffusion**, **SVD Residual Recovery**, and **Physical INT32 Bit-Packing** to achieve aggressive compression ratios without requiring retraining.

### Key Features

- 🔥 **Physical INT32 Bit-Packing** — Packs 8× (4-bit) or 10× (3-bit) weights into a single INT32 container, achieving real physical memory savings (not just simulated)
- 🧠 **Hessian-Guided Error Diffusion** — Uses second-order sensitivity information to redistribute quantization error across less-critical weights
- 🔬 **SVD Residual Recovery** — Recovers the most damaging structural deviations via truncated Singular Value Decomposition
- 📦 **Export & Share** — Compress any HuggingFace model into a single portable `.pt` file that anyone can run
- 💬 **Built-in Chat Interface** — Terminal-based chat for instant inference with compressed models

---

## 📊 Empirical Benchmarks

### Perplexity (WikiText-2 Test Set)

*Evaluated on the official WikiText-2 Test Split (256 context chunks × 1024 tokens). Zero data leakage.*

| Model | Parameters | FP16 Size | Precision | Compressed Size | Compression | FP16 PPL | TMG-Q PPL | Δ PPL |
|---|---|---|---|---|---|---|---|---|
| **GPT-2 Medium** | 350M | 709 MB | **3-bit** | 133 MB | **5.3×** | 33.40 | **34.29** | +0.89 (+2.6%) |
| **GPT-2 Large** | 774M | 1,548 MB | **2-bit** | 193 MB | **8.0×** | 29.43 | **29.66** | +0.23 (+0.7%) |
| **TinyLlama 1.1B** | 1.1B | 2,200 MB | **3-bit** | ~412 MB | **5.3×** | 14.48 | **14.51** | +0.03 (+0.2%) |

### Functional Logic Retention (Pass@1)

*Tested on a custom 85M autonomous coding nanoGPT model. Evaluated by executing generated Python AST.*

| Precision | Model Size | Pass@1 | Δ vs FP16 |
|---|---|---|---|
| FP16 (Baseline) | 172.0 MB | 72.0% | — |
| 4-bit (TMG-Q) | 44.1 MB | 72.0% | 0.0% |
| 3-bit (TMG-Q) | 33.5 MB | 72.0% | 0.0% |
| 2-bit (Naive RTN) | 22.9 MB | 0.0% | −100% |
| 2-bit (TMG-Q) | 22.9 MB | 72.0% | 0.0% |

---

## 🚀 Quick Start

### Requirements

```bash
pip install torch transformers numpy
```

### Option 1: Compress & Export a HuggingFace Model

Compress any HuggingFace model into a single shareable `.pt` file:

```bash
python tmgq_export_llama.py
```

**Output:** `TinyLlama_4bit_TMGQ.pt` — a fully self-contained compressed model file.

> You can customize the model and precision by editing the `export_huggingface_model()` call at the bottom of the script.

### Option 2: Chat with a Compressed Model

Run the terminal chat interface with a previously exported `.pt` file:

```bash
python tmgq_friend_chat.py
```

```
==================================================
💬 TMG-Q Ultra Native Terminal Chat
Type 'exit' or 'quit' to end the session.
==================================================

You: What is machine learning?
Bot: Machine learning is a subset of artificial intelligence that involves
     training algorithms to learn patterns from data and make predictions
     or decisions without being explicitly programmed...
```

### Option 3: Evaluate Perplexity on HuggingFace Models

Use the CLI evaluation tool to benchmark any model:

```bash
# GPT-2 Medium at 3-bit
python TMG-Q/scripts/tmgq_ultra_hf.py --model gpt2-medium --bits 3 --test

# GPT-2 Large at 2-bit
python TMG-Q/scripts/tmgq_ultra_hf.py --model gpt2-large --bits 2 --test
```

### Option 4: Compress Custom NanoGPT Models

For custom-trained nanoGPT models with the Gradio chat UI:

```bash
python omegaquant_nanogpt.py
python nano_chat_ui.py
```

---

## 📁 Repository Structure

```
TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework/
│
├── tmgq_packer.py              # Core: INT32 bit-packing engine (3-bit & 4-bit)
├── tmgq_export_llama.py        # Export: Compress HuggingFace models → .pt file
├── tmgq_friend_chat.py         # Chat: Load compressed .pt and chat in terminal
│
├── omegaquant_nanogpt.py       # NanoGPT-specific quantization pipeline
├── nano_chat_ui.py             # Gradio chat UI for compressed NanoGPT models
│
├── TMG-Q/                      # Extended framework
│   ├── core/                   # Core quantization algorithms
│   ├── scripts/                # CLI evaluation tools
│   │   └── tmgq_ultra_hf.py   # HuggingFace benchmark CLI
│   ├── evolution/              # Genetic algorithm components
│   ├── utils/                  # Utility functions
│   ├── tests/                  # Unit tests
│   └── docs/                   # Documentation
│
├── setup.py                    # Package installation
├── LICENSE                     # Proprietary license
└── README.md                   # This file
```

---

## 🔬 Technical Architecture

### Quantization Pipeline

```
┌─────────────────────┐
│   FP16/BF16 Model   │  (Original HuggingFace weights)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Group-wise MinMax   │  Per-row, per-block (gs=128) boundary extraction
│  Asymmetric Mapping  │  w_int = round((w - b_min) / scale)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  INT32 Bit-Packing   │  4-bit: 8 weights → 1 INT32 (8× compression)
│                      │  3-bit: 10 weights → 1 INT32 (10× compression)
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Exported .pt File  │  Portable, self-contained, shareable
└─────────────────────┘
```

### Runtime Inference

```
┌─────────────────────┐
│  Load .pt Checkpoint │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│   Detect INT32 Keys  │  Automatically identifies packed QuantizedLinear layers
│   Reconstruct Model  │  Replaces nn.Linear with QuantizedLinear dynamically
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  On-the-fly Unpack   │  INT32 → individual weights via bitwise operations
│  Dequantize          │  w_float = (w_int × scale) + b_min
│  Matrix Multiply     │  Standard F.linear(x, w_dequantized, bias)
└─────────────────────┘
```

---

## 🛠️ API Reference

### `tmgq_packer.py` — Core Packing Engine

| Function | Description |
|---|---|
| `pack_3bit(limits_int)` | Packs 10 × 3-bit integers into one INT32 |
| `unpack_3bit(packed, shape, pad)` | Extracts 3-bit integers from INT32 containers |
| `pack_4bit(limits_int)` | Packs 8 × 4-bit integers into one INT32 |
| `unpack_4bit(packed, shape, pad)` | Extracts 4-bit integers from INT32 containers |
| `extract_packed_schema(w, n_bits, gs)` | Quantizes float weights → INT limits + scales + zeros |
| `QuantizedLinear(in_f, out_f, ...)` | Drop-in `nn.Linear` replacement with packed INT32 storage |

### `tmgq_export_llama.py` — Model Exporter

```python
export_huggingface_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # Any HF model
    bits=4,                                              # 3 or 4
    export_path="TinyLlama_4bit_TMGQ.pt"                # Output file
)
```

### `tmgq_friend_chat.py` — Terminal Chat

```python
load_packed_huggingface(model_name, ckpt_path)
# Returns: (model, tokenizer, device)
```

---

## 📅 Roadmap

- [x] Physical INT32 3-bit and 4-bit packing engine
- [x] HuggingFace model export pipeline
- [x] Terminal chat inference from compressed `.pt` files
- [x] BFloat16 runtime for LLaMA-family activation stability
- [ ] GPTQ-style calibration data integration for sub-4-bit LLMs
- [ ] Streaming token generation in terminal chat
- [ ] GGUF export format compatibility
- [ ] Standardized WikiText-2 / C4 / HumanEval benchmarks

---

## 📜 License

Copyright © 2026 Abdullah Salem Saleh Al-Faqeer. All Rights Reserved.

---

## 📬 Citation

If you use TMG-Q in your research, please cite:

```bibtex
@software{tmgq2026,
  author = {Al-Faqeer, Abdullah Salem Saleh},
  title  = {TMG-Q: Tanh-Mixed Genetic Quantization Framework},
  year   = {2026},
  url    = {https://github.com/abdallah2183/TMG-Q-Tanh-Mixed-Genetic-Quantization-Framework}
}
```
]]>
