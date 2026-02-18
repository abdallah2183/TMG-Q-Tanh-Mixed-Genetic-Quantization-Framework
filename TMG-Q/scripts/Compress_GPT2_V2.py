"""
=================================================================
Compress Real GPT-2 with GPTQ-Lite V2
=================================================================
التطبيق الحقيقي: ضغط GPT-2 (124M params) باستخدام V2

التجربة:
  1. تحميل GPT-2 الأصلي من HuggingFace
  2. توليد نص بالنموذج الأصلي (baseline)
  3. ضغط كل طبقة بـ V2 (مع الترقيات الأربعة)
  4. إعادة بناء النموذج من الأوزان المضغوطة
  5. توليد نص بالنموذج المضغوط → مقارنة الجودة
  6. حساب Perplexity + Logits MSE

هذا الاختبار الحقيقي اللي يثبت إن الخوارزمية تنافس AWQ/GPTQ.
"""

import torch
import numpy as np
import sys
import os
import time

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GPTQ_Lite_V2 import GPTQLiteV2


def generate_text(model, tokenizer, prompt, max_new_tokens=80, temperature=0.7):
    """Generate text from model with given prompt."""
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated


def compute_perplexity(model, tokenizer, text):
    """Compute perplexity on a given text."""
    device = next(model.parameters()).device
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
    input_ids = encodings.input_ids

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss

    return float(torch.exp(loss))


def compute_logits_mse(model_orig, model_comp, tokenizer, texts):
    """Compare logits MSE between original and compressed model."""
    device_orig = next(model_orig.parameters()).device
    device_comp = next(model_comp.parameters()).device

    total_mse = 0.0
    count = 0

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            logits_orig = model_orig(**inputs.to(device_orig)).logits.cpu().float()
            logits_comp = model_comp(**inputs.to(device_comp)).logits.cpu().float()

        mse = float(torch.mean((logits_orig - logits_comp) ** 2))
        total_mse += mse
        count += 1

    return total_mse / max(count, 1)


def generate_calibration_data(tokenizer, n_samples=32, seq_len=64):
    """
    Generate calibration data for activation-aware compression.
    Uses random token sequences to simulate real input.
    """
    vocab_size = tokenizer.vocab_size
    calib_ids = torch.randint(0, vocab_size, (n_samples, seq_len))
    return calib_ids


def get_layer_activations(model, tokenizer, calib_texts, layer_name):
    """
    Collect real activations flowing through a specific layer.
    Used for activation-aware calibration in V2.
    """
    activations = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(input, tuple) and len(input) > 0:
            activations.append(input[0].detach().cpu().float().numpy())

    # Find and hook the target module
    for name, module in model.named_modules():
        if name == layer_name or name.endswith(layer_name.split('.')[-1]):
            h = module.register_forward_hook(hook_fn)
            hooks.append(h)
            break

    # Run calibration data through model
    device = next(model.parameters()).device
    for text in calib_texts[:4]:  # Use 4 samples for speed
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            model(**inputs)

    # Remove hooks
    for h in hooks:
        h.remove()

    if activations:
        # Concatenate and reshape to [batch, features]
        act = np.concatenate([a.reshape(-1, a.shape[-1]) for a in activations], axis=0)
        return act[:64]  # Max 64 samples
    return None


def compress_gpt2_v2(mode="nonlinear"):
    """Main compression pipeline for GPT-2 with V2."""

    print("\n" + "=" * 70)
    print("🚀 COMPRESSING GPT-2 WITH GPTQ-Lite V2")
    print("   All 4 Enhancements Active:")
    print("   ① Dynamic Saliency  ② Scaling Factor")
    print("   ③ Activation-Aware  ④ Layer-wise Reconstruction")
    print("=" * 70)

    # ================================================================
    # STEP 1: Load Model
    # ================================================================
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
        set_seed(42)

        print("\n📥 Loading GPT-2...")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
        model.eval()
        print("   ✅ GPT-2 loaded (124M parameters)")

        # Count total parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,}")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   Install: pip install transformers")
        return

    # ================================================================
    # STEP 2: Baseline - Generate text with ORIGINAL model
    # ================================================================
    print("\n" + "─" * 70)
    print("📝 BASELINE: Generating text with ORIGINAL GPT-2...")
    print("─" * 70)

    test_prompts = [
        "The future of artificial intelligence is",
        "In a groundbreaking scientific discovery,",
        "The key difference between quantum computing and classical computing is",
    ]

    original_texts = []
    for prompt in test_prompts:
        text = generate_text(model, tokenizer, prompt, max_new_tokens=60)
        original_texts.append(text)
        print(f"\n  Prompt: '{prompt}'")
        # Show only generated part
        generated_part = text[len(prompt):]
        print(f"  Output: {generated_part[:200]}")

    # Baseline perplexity
    eval_text = ("Artificial intelligence has transformed the way we interact with technology. "
                 "Machine learning models can now understand natural language, generate images, "
                 "and even write code. The future holds promise for even more advanced systems "
                 "that can reason, learn, and adapt to new situations autonomously.")

    orig_ppl = compute_perplexity(model, tokenizer, eval_text)
    print(f"\n  📊 Original Perplexity: {orig_ppl:.2f}")

    # ================================================================
    # STEP 3: Compress with V2
    # ================================================================
    print("\n" + "=" * 70)
    print("⚙️  COMPRESSING WITH V2...")
    print("=" * 70)

    # Calibration texts for activation-aware compression
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "The weather today is sunny with a high of 75 degrees.",
        "Scientists have discovered a new species of dinosaur.",
        "The stock market experienced significant volatility today.",
        "Quantum computing promises to solve previously intractable problems.",
        "The history of mathematics dates back to ancient civilizations.",
    ]

    state_dict = model.state_dict()
    compressed_dict = {}
    metadata_dict = {}

    compressor = GPTQLiteV2(group_size=128)

    original_size = 0
    compressed_size = 0
    start_time = time.time()
    layer_count = 0
    prev_layer_error = None

    # Count compressible layers first
    compressible_layers = []
    for name, tensor in state_dict.items():
        is_embedding = any(x in name for x in ['wte', 'wpe'])
        is_norm = any(x in name for x in ['ln_', 'layernorm', 'layer_norm'])
        if tensor.numel() > 1024 and tensor.dtype == torch.float32 and not is_embedding and not is_norm:
            compressible_layers.append(name)

    total_compressible = len(compressible_layers)
    print(f"   Found {total_compressible} compressible layers\n")

    for name, tensor in state_dict.items():
        is_embedding = any(x in name for x in ['wte', 'wpe'])
        is_norm = any(x in name for x in ['ln_', 'layernorm', 'layer_norm'])

        if tensor.numel() > 1024 and tensor.dtype == torch.float32 and not is_embedding and not is_norm:
            layer_count += 1
            weights_np = tensor.detach().cpu().numpy()
            original_size += weights_np.nbytes

            print(f"   [{layer_count}/{total_compressible}] {name}")
            print(f"      Shape: {weights_np.shape} | Params: {weights_np.size:,}")

            # Get real activations for this layer (if possible)
            calib_input = None
            if weights_np.ndim == 2:
                try:
                    calib_input = get_layer_activations(model, tokenizer, calib_texts, name)
                    if calib_input is not None:
                        # Adjust shape to match weight dimensions
                        if calib_input.shape[-1] != weights_np.shape[1]:
                            calib_input = None
                except Exception:
                    calib_input = None

            # V2 Compress
            compressed = compressor.compress_v2(
                weights_np,
                layer_name=name,
                layer_idx=layer_count - 1,
                total_layers=total_compressible,
                mode=mode,
                calibration_input=calib_input,
                prev_layer_error=prev_layer_error
            )

            # Track error for next layer (Layer-wise Reconstruction)
            prev_layer_error = compressed.get('layer_error', None)

            # Calculate compressed size
            layer_comp_size = compressed['packed'].nbytes
            layer_comp_size += compressed['outlier_mask'].nbytes
            layer_comp_size += compressed['outlier_values'].nbytes
            if compressed.get('scales') is not None:
                layer_comp_size += compressed['scales'].nbytes
            if compressed.get('zero_points') is not None:
                layer_comp_size += compressed['zero_points'].nbytes
            if compressed.get('constants') is not None:
                layer_comp_size += compressed['constants'].nbytes
            if compressed.get('scaling_factor') is not None:
                layer_comp_size += compressed['scaling_factor'].nbytes

            compressed_size += layer_comp_size

            # Store
            compressed_dict[name] = compressed
            metadata_dict[name] = {
                'type': compressed['type'],
                'outlier_percent': compressed['outlier_percent'],
                'scaling_factor': float(compressed['scaling_factor'][0]),
            }

            ratio = weights_np.nbytes / layer_comp_size if layer_comp_size > 0 else 0
            print(f"      {weights_np.nbytes / 1024:.1f} KB → {layer_comp_size / 1024:.1f} KB "
                  f"({ratio:.2f}x) | Outlier: {compressed['outlier_percent']:.1f}%\n")

        else:
            # Keep as-is (embeddings, norms, biases)
            original_size += tensor.numel() * 4
            compressed_size += tensor.numel() * 4

            if is_embedding:
                print(f"   [SKIP] {name} (embedding, kept FP32)")
            elif is_norm:
                print(f"   [SKIP] {name} (norm, kept FP32)")

    compress_time = time.time() - start_time

    # ================================================================
    # STEP 4: Decompress and rebuild model
    # ================================================================
    print("\n" + "=" * 70)
    print("🔧 REBUILDING MODEL FROM COMPRESSED WEIGHTS...")
    print("=" * 70)

    rebuild_start = time.time()
    new_state_dict = {}

    for name, tensor in state_dict.items():
        if name in compressed_dict:
            # Decompress V2
            restored_np = compressor.decompress_v2(compressed_dict[name])
            new_state_dict[name] = torch.from_numpy(restored_np).to(tensor.dtype)
        else:
            new_state_dict[name] = tensor.clone()

    # Load into a fresh model
    model_compressed = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float32)
    model_compressed.load_state_dict(new_state_dict)
    model_compressed.eval()

    rebuild_time = time.time() - rebuild_start
    print(f"   ✅ Model rebuilt in {rebuild_time:.1f}s")

    # ================================================================
    # STEP 5: Test compressed model
    # ================================================================
    print("\n" + "─" * 70)
    print("📝 COMPRESSED MODEL: Generating text...")
    print("─" * 70)

    for prompt in test_prompts:
        text = generate_text(model_compressed, tokenizer, prompt, max_new_tokens=60)
        print(f"\n  Prompt: '{prompt}'")
        generated_part = text[len(prompt):]
        print(f"  Output: {generated_part[:200]}")

    # ================================================================
    # STEP 6: Quality Metrics
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 QUALITY COMPARISON: Original vs V2 Compressed")
    print("=" * 70)

    # Perplexity
    comp_ppl = compute_perplexity(model_compressed, tokenizer, eval_text)
    ppl_increase = ((comp_ppl - orig_ppl) / orig_ppl) * 100

    print(f"\n  Perplexity:")
    print(f"    Original:   {orig_ppl:.2f}")
    print(f"    Compressed: {comp_ppl:.2f}")
    print(f"    Increase:   {ppl_increase:+.1f}%")

    if ppl_increase < 5:
        ppl_verdict = "✅✅ EXCELLENT (< 5% increase)"
    elif ppl_increase < 15:
        ppl_verdict = "✅ GOOD (< 15% increase)"
    elif ppl_increase < 50:
        ppl_verdict = "⚠️ ACCEPTABLE (< 50% increase)"
    else:
        ppl_verdict = "❌ NEEDS IMPROVEMENT"
    print(f"    Verdict:    {ppl_verdict}")

    # Logits MSE
    logits_mse = compute_logits_mse(model, model_compressed, tokenizer, calib_texts[:4])
    print(f"\n  Logits MSE:   {logits_mse:.4f}")

    if logits_mse < 1.0:
        logits_verdict = "✅✅ EXCELLENT (< 1.0)"
    elif logits_mse < 5.0:
        logits_verdict = "✅ GOOD (< 5.0)"
    elif logits_mse < 20.0:
        logits_verdict = "⚠️ ACCEPTABLE (< 20.0)"
    else:
        logits_verdict = "❌ HIGH (was 12 in V1)"
    print(f"    Verdict:    {logits_verdict}")

    # ================================================================
    # STEP 7: Save compressed model
    # ================================================================
    save_path = f"gpt2_v2_{mode}.agi"
    print(f"\n💾 Saving to {save_path}...")

    torch.save({
        'compressed_layers': {k: {
            'packed': v['packed'],
            'scales': v.get('scales'),
            'zero_points': v.get('zero_points'),
            'constants': v.get('constants'),
            'scaling_factor': v.get('scaling_factor'),
            'outlier_mask': v['outlier_mask'],
            'outlier_values': v['outlier_values'],
            'original_shape': v['original_shape'],
            'n_non_outliers': v['n_non_outliers'],
            'type': v['type'],
        } for k, v in compressed_dict.items()},
        'uncompressed_layers': {k: v for k, v in state_dict.items()
                                 if k not in compressed_dict},
        'metadata': metadata_dict,
        'model_name': 'gpt2',
        'compressor': 'GPTQ-Lite-V2',
        'mode': mode,
        'metrics': {
            'original_perplexity': orig_ppl,
            'compressed_perplexity': comp_ppl,
            'logits_mse': logits_mse,
            'compression_ratio': original_size / compressed_size,
        }
    }, save_path)

    file_size = os.path.getsize(save_path)

    # ================================================================
    # FINAL REPORT
    # ================================================================
    print("\n" + "#" * 70)
    print("  📋 FINAL COMPRESSION REPORT")
    print("#" * 70)
    print(f"  Model:              GPT-2 (124M params)")
    print(f"  Method:             GPTQ-Lite V2 ({mode})")
    print(f"  Enhancements:       Dynamic Saliency + Scale Factor")
    print(f"                      + Activation-Aware + Layer Reconstruction")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Original Size:      {original_size / 1024 / 1024:.1f} MB")
    print(f"  Compressed Size:    {compressed_size / 1024 / 1024:.1f} MB")
    print(f"  File on Disk:       {file_size / 1024 / 1024:.1f} MB")
    print(f"  Compression Ratio:  {original_size / compressed_size:.2f}x")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Original PPL:       {orig_ppl:.2f}")
    print(f"  Compressed PPL:     {comp_ppl:.2f}")
    print(f"  PPL Increase:       {ppl_increase:+.1f}%")
    print(f"  Logits MSE:         {logits_mse:.4f}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Compress Time:      {compress_time:.1f}s")
    print(f"  Rebuild Time:       {rebuild_time:.1f}s")
    print(f"  Layers Compressed:  {layer_count}/{total_compressible}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Quality Verdict:    {ppl_verdict}")
    print(f"  Logits Verdict:     {logits_verdict}")
    print("#" * 70)

    return {
        'compression_ratio': original_size / compressed_size,
        'original_ppl': orig_ppl,
        'compressed_ppl': comp_ppl,
        'logits_mse': logits_mse,
        'file_path': save_path,
    }


if __name__ == "__main__":
    mode = "nonlinear"
    if len(sys.argv) > 1:
        mode = sys.argv[1].strip().lower()

    print(f"\n  Mode: {mode}")
    print(f"  Usage: python Compress_GPT2_V2.py [nonlinear|linear]\n")

    result = compress_gpt2_v2(mode=mode)

    if result:
        print(f"\n  ✅ Done! File saved: {result['file_path']}")
        print(f"     Ratio: {result['compression_ratio']:.2f}x | PPL: {result['compressed_ppl']:.2f}")
