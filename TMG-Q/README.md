# TMG-Q: Tanh-Nonlinear Mixed-Precision Genetic Quantization

<p align="center">
  <strong>A Novel Post-Training Quantization Framework for Large Language Models</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0%2B-red?logo=pytorch" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-All%20Rights%20Reserved-red" alt="License">
  <img src="https://img.shields.io/badge/GPU-Not%20Required-brightgreen" alt="No GPU">
  <img src="https://img.shields.io/badge/Status-Experimental-orange" alt="Status">
</p>

---

## Highlights

- **2.01x compression** on GPT-2 Medium (355M) with only **+2.5% perplexity**
- **99.74% cosine similarity** on LLaVA-7B weight reconstruction
- **No GPU required** — entire compression pipeline runs on CPU
- **Genetically discovered** quantization formulas via HyperEvolution
- **Tanh-based nonlinear quantization** — 4.2x more precision near zero than linear methods

---

## What is TMG-Q?

TMG-Q (**T**anh-**M**ixed-**G**enetic **Q**uantization) is a post-training quantization framework that compresses LLM weights from FP32 to INT4 while preserving model quality. Unlike existing methods (GPTQ, AWQ, GGUF), TMG-Q uses:

| Component | Description |
|-----------|-------------|
| **T — Tanh** | Nonlinear quantization function discovered through genetic programming. Provides higher resolution near zero where most LLM weights concentrate. |
| **M — Mixed** | Sensitivity-driven mixed precision: FP32 for embeddings/norms, FP16 for boundary layers, INT4 for middle blocks. |
| **G — Genetic** | HyperEvolution algorithm evolves compression formulas over 15,000+ generations using activation-aware fitness. |
| **Q — Quantization** | Four enhancements: dynamic outlier saliency, least-squares scaling, activation-aware calibration, cross-layer error reconstruction. |

---

## Project Structure

```
TMG-Q/
│
├── core/                               # Core Algorithm
│   ├── TMG_Q.py                        # Public API — start here
│   ├── GPTQ_Lite_V2.py                 # Core quantization engine (V2)
│   └── GPTQ_Lite.py                    # V1 baseline engine
│
├── evolution/                          # Genetic Algorithm
│   ├── HyperEvolution.py              # V1 — weight MSE fitness
│   └── HyperEvolution_V2.py           # V2 — activation-aware fitness
│
├── scripts/                            # Compression Scripts
│   ├── Chat_GPT2_V2.py                # Compress GPT-2 + interactive chat
│   ├── Compress_GPT2_V2.py            # GPT-2 Medium compression
│   ├── Compress_GPT2_V2_Refined.py    # GPT-2 refined layer strategy
│   ├── Compress_Llama3_V2.py          # LLaMA-3 8B compression
│   └── Compress_Ollama_V2.py          # LLaVA-7B GGUF compression
│
├── tests/                              # Quality Testing
│   └── Test_V2_Quality.py             # Comprehensive quality test suite
│
├── utils/                              # Utilities
│   └── math_utils.py                  # Mathematical helper functions
│
├── docs/                               # Documentation
│   └── TMG_Q_Research_Paper.md        # Full research paper
│
├── README.md                           # This file
├── LICENSE                             # All Rights Reserved
└── requirements.txt                    # Python dependencies
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/abdal/TMG-Q.git
cd TMG-Q
pip install -r requirements.txt
```

### Basic Usage

```python
import sys
sys.path.insert(0, 'core')
from TMG_Q import TMGQ

# Initialize compressor
compressor = TMGQ(group_size=128)
compressor.info()

# Compress a weight matrix
result = compressor.compress(
    weights,                    # numpy float32 [out_features, in_features]
    layer_name="model.h.5.mlp.c_fc.weight",
    layer_idx=5,
    total_layers=24,
    mode='linear',
    calibration_input=calib,    # optional
    prev_layer_error=prev_err   # optional
)

# Decompress
restored = compressor.decompress(result)
```

### Compress GPT-2 Medium & Chat

```bash
cd scripts
python Chat_GPT2_V2.py
```

This will:
1. Download GPT-2 Medium (~1.4GB)
2. Compress with TMG-Q (~43 seconds on CPU)
3. Report compression ratio and perplexity
4. Launch interactive chat with the compressed model

### Compress LLaVA-7B from Ollama

```bash
# First pull the model
ollama pull llava

# Then compress and compare
cd scripts
python Compress_Ollama_V2.py
```

### Run Quality Tests

```bash
cd tests
python Test_V2_Quality.py
```

### Discover New Formulas with HyperEvolution

```bash
cd evolution
python HyperEvolution_V2.py
```

---

## Results

### GPT-2 Medium (355M Parameters)

| Metric | Value |
|--------|-------|
| Original Size | 1,354 MB (FP32) |
| Compressed Size | 770 MB |
| Compression Ratio | **2.01x** |
| Perplexity (FP32 baseline) | 17.67 |
| Perplexity (TMG-Q) | 18.11 (**+2.5%**) |
| Compression Time | 43 seconds (CPU only) |

### LLaVA-7B Weight Reconstruction

| Metric | Average |
|--------|---------|
| MSE | 0.000020 |
| Cosine Similarity | **0.9971** |
| SNR | 39.5 dB |

### Ablation: Impact of Each Enhancement

| Enhancement | Perplexity Increase |
|------------|-------------------|
| V1 baseline (uniform INT4) | +120-150% |
| + Mixed-precision layers | +25-40% |
| + Dynamic saliency | +15-25% |
| + Scaling factor | +8-15% |
| + Activation-aware calibration | +3-8% |
| + Layer-wise reconstruction (full TMG-Q) | **+2.5%** |

---

## The TMG-Q Algorithm

### Tanh Quantization Formula

**Compress:**
$$q = \frac{w}{|\tanh(c)| + \frac{w}{w + \text{sign}(w) \cdot \epsilon} - |c|}$$

**Decompress:**
$$\hat{w} = q \cdot (\tanh(c) - c) + q$$

This formula was **discovered by genetic programming**, not hand-designed. The tanh nonlinearity provides 4.2x more quantization levels in the [-0.1, 0.1] range where ~40% of LLM weights reside.

### Mixed-Precision Strategy

| Precision | Layers | Rationale |
|-----------|--------|-----------|
| FP32 (skip) | Embeddings, norms, biases | Tiny but critical; errors propagate everywhere |
| FP16 (2x) | lm_head, first/last 2 blocks | Boundary layers most sensitive to quantization |
| INT4 (~8x) | All middle transformer blocks | Bulk of parameters; most tolerant to compression |

### Four Enhancements

1. **Dynamic Outlier Saliency** — Adaptive 1-10% outlier protection per layer based on position sensitivity
2. **Per-Layer Scaling Factor** — Least-squares optimal scaling to correct magnitude drift
3. **Activation-Aware Calibration** — 70% activation MSE + 30% weight MSE scoring
4. **Layer-wise Error Reconstruction** — Cross-layer error compensation to prevent error snowballing

---

## Comparison with Existing Methods

| Feature | GPTQ | AWQ | GGUF Q4_0 | TMG-Q (ours) |
|---------|------|-----|-----------|--------------|
| Quantization | Linear | Linear | Linear | **Tanh nonlinear** |
| Formula origin | Hand-designed | Hand-designed | Hand-designed | **Genetically evolved** |
| Precision | Uniform W4 | Uniform W4 | Uniform Q4 | **Mixed FP32/16/4** |
| GPU required | Yes | Yes | No | **No** |
| Calibration | Required | Required | None | **Optional** |
| Error propagation | No | No | No | **Yes** |
| Outlier handling | Column-order | 1% salient | None | **Dynamic 1-10%** |

---

## Requirements

```
Python >= 3.10
NumPy >= 1.24
PyTorch >= 2.0
Transformers >= 4.35
gguf >= 0.6.0 (optional, for Ollama models)
```

---

## Research Paper

The full research paper with theoretical analysis, proofs, and detailed experimental results is available at:

**[docs/TMG_Q_Research_Paper.md](docs/TMG_Q_Research_Paper.md)**

---

## Citation

```bibtex
@misc{abdal2026tmgq,
  title     = {TMG-Q: Tanh-Nonlinear Mixed-Precision Genetic Quantization
               for High-Quality LLM Compression},
  author    = {Abdal},
  year      = {2026},
  url       = {https://github.com/abdal/TMG-Q}
}
```

---

## License

**Copyright (c) 2026 Abdal. All Rights Reserved.**

This software is proprietary. You may NOT use, copy, modify, or distribute it without prior written permission from the author. See [LICENSE](LICENSE) for full terms.

---

**Made with science and evolution by Abdal**
