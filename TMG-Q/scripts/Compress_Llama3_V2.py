"""
=================================================================
Compress Llama-3 8B with GPTQ-Lite V2
=================================================================
ضغط نموذج Llama-3 8B (8 مليار parameter) باستخدام V2

استراتيجية الذاكرة:
  - تحميل النموذج بـ FP16 مع device_map="cpu"
  - ضغط طبقة بطبقة (لا نحتاج GPU)
  - حذف الأوزان الأصلية بعد الضغط (تحرير RAM)
  - اختبار الجودة على عينات صغيرة

متطلبات:
  pip install transformers accelerate safetensors
  
  Llama-3 يحتاج HuggingFace token:
  huggingface-cli login
  
  أو يمكن استخدام نموذج بديل بدون token.
"""

import torch
import numpy as np
import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from GPTQ_Lite_V2 import GPTQLiteV2


# ================================================================
# Memory utilities
# ================================================================
def get_ram_usage():
    import psutil
    m = psutil.virtual_memory()
    return m.used / 1024**3, m.available / 1024**3, m.total / 1024**3

def print_ram():
    used, avail, total = get_ram_usage()
    print(f"   💾 RAM: {used:.1f}GB used / {avail:.1f}GB free / {total:.1f}GB total")


# ================================================================
# Layer classification for Llama architecture
# ================================================================
def classify_llama_layer(name, layer_idx, total_layers):
    """
    Smart layer classification for Llama-3 architecture.
    
    Llama layer naming:
      model.embed_tokens.weight
      model.layers.0.self_attn.q_proj.weight
      model.layers.0.self_attn.k_proj.weight
      model.layers.0.self_attn.v_proj.weight
      model.layers.0.self_attn.o_proj.weight
      model.layers.0.mlp.gate_proj.weight
      model.layers.0.mlp.up_proj.weight
      model.layers.0.mlp.down_proj.weight
      model.layers.0.input_layernorm.weight
      model.layers.0.post_attention_layernorm.weight
      model.norm.weight
      lm_head.weight
    """
    n = name.lower()

    # NEVER compress
    if 'embed_tokens' in n:
        return 'skip'
    if 'layernorm' in n or 'layer_norm' in n or (n == 'model.norm.weight'):
        return 'skip'
    if 'rotary' in n or 'inv_freq' in n:
        return 'skip'

    # lm_head → FP16 (critical for output quality)
    if 'lm_head' in n:
        return 'fp16'

    # Extract layer number
    import re
    m = re.search(r'layers\.(\d+)\.', n)
    if m:
        layer_num = int(m.group(1))
        # First 2 and last 2 layers → FP16
        if layer_num <= 1 or layer_num >= (total_layers - 2):
            return 'fp16'

    # Everything else → INT4 with V2
    return 'linear'


def count_llama_layers(state_dict):
    """Count how many transformer layers exist."""
    import re
    max_layer = -1
    for name in state_dict.keys():
        m = re.search(r'layers\.(\d+)\.', name)
        if m:
            max_layer = max(max_layer, int(m.group(1)))
    return max_layer + 1 if max_layer >= 0 else 0


# ================================================================
# Model loading strategies
# ================================================================
def try_load_model(model_id):
    """Try to load model with various fallback strategies."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

    print(f"\n📥 Attempting to load: {model_id}")
    print_ram()

    try:
        print("   Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        print("   Loading config...")
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        print(f"   Architecture: {config.architectures}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Num layers: {config.num_hidden_layers}")
        print(f"   Vocab size: {config.vocab_size}")

        print("   Loading model (FP16, CPU)...")
        print("   ⏳ This may take several minutes for 8B models...")
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                device_map="cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
        except Exception:
            # Fallback: load without device_map (no accelerate needed)
            print("   (Falling back to direct CPU load...)")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            )
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ Loaded! {total_params:,} parameters")
        print_ram()

        return model, tokenizer, config

    except Exception as e:
        print(f"   ❌ Failed: {e}")
        return None, None, None


# ================================================================
# Text generation (minimal, CPU-friendly)
# ================================================================
def generate_text_cpu(model, tokenizer, prompt, max_new_tokens=50):
    """Generate text on CPU (slow but works without GPU)."""
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            top_k=50, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def compute_perplexity_cpu(model, tokenizer, text):
    """Compute perplexity on CPU."""
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        loss = model(enc.input_ids, labels=enc.input_ids).loss
    return float(torch.exp(loss))


# ================================================================
# Main compression pipeline
# ================================================================
def compress_llama3(model_id=None):
    """
    Full Llama-3 8B compression pipeline.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\n" + "=" * 70)
    print("🦙 COMPRESSING LLAMA-3 8B WITH GPTQ-Lite V2")
    print("   All 4 Enhancements Active:")
    print("   ① Dynamic Saliency  ② Scaling Factor")
    print("   ③ Activation-Aware  ④ Layer-wise Reconstruction")
    print("=" * 70)
    print_ram()

    # ================================================================
    # Step 1: Try to load model (with fallbacks)
    # ================================================================
    model_candidates = []

    if model_id:
        model_candidates.append(model_id)

    model_candidates.extend([
        # Llama-3 8B (needs Meta approval + HF token)
        "meta-llama/Meta-Llama-3-8B",
        # Llama-3.1 8B
        "meta-llama/Llama-3.1-8B",
        # Llama-3.2 3B (smaller, easier to load)
        "meta-llama/Llama-3.2-3B",
        # Open alternatives (no token needed)
        "microsoft/phi-2",              # 2.7B, very good quality
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B fallback
    ])

    model = None
    tokenizer = None
    config = None
    actual_model_id = None

    for mid in model_candidates:
        print(f"\n{'─' * 50}")
        model, tokenizer, config = try_load_model(mid)
        if model is not None:
            actual_model_id = mid
            break

    if model is None:
        print("\n❌ Could not load any model.")
        print("   Try one of these:")
        print("   1. pip install transformers accelerate")
        print("   2. huggingface-cli login  (for Llama-3)")
        print("   3. Ensure you have enough RAM (16GB+ for 8B models)")
        return

    total_params = sum(p.numel() for p in model.parameters())
    model_size_gb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3

    print(f"\n   📊 Model: {actual_model_id}")
    print(f"   Parameters: {total_params:,}")
    print(f"   Model size (in memory): {model_size_gb:.2f} GB")

    # ================================================================
    # Step 2: Baseline evaluation
    # ================================================================
    print("\n" + "─" * 70)
    print("📝 BASELINE: Original model text generation")
    print("─" * 70)

    test_prompts = [
        "The future of artificial intelligence is",
        "In a groundbreaking scientific discovery,",
    ]

    print("   ⏳ Generating (CPU, may take 30-60s per prompt)...")
    for prompt in test_prompts:
        try:
            text = generate_text_cpu(model, tokenizer, prompt, max_new_tokens=40)
            print(f"\n  Prompt: '{prompt}'")
            print(f"  Output: {text[len(prompt):][:200]}")
        except Exception as e:
            print(f"  ⚠️ Generation failed: {e}")

    eval_text = ("Artificial intelligence has transformed the way we interact with technology. "
                 "Machine learning models can now understand natural language and generate text.")

    print("\n   Computing baseline perplexity...")
    try:
        orig_ppl = compute_perplexity_cpu(model, tokenizer, eval_text)
        print(f"   📊 Original Perplexity: {orig_ppl:.2f}")
    except Exception as e:
        orig_ppl = None
        print(f"   ⚠️ PPL failed: {e}")

    # ================================================================
    # Step 3: Compress layer by layer
    # ================================================================
    print("\n" + "=" * 70)
    print("⚙️  COMPRESSING WITH V2...")
    print("=" * 70)

    state_dict = model.state_dict()
    n_transformer_layers = count_llama_layers(state_dict)
    print(f"   Transformer layers: {n_transformer_layers}")

    # Classify all layers
    layer_strategies = {}
    strategy_counts = {'skip': 0, 'fp16': 0, 'linear': 0, 'nonlinear': 0}

    for name in state_dict.keys():
        strategy = classify_llama_layer(name, 0, n_transformer_layers)
        layer_strategies[name] = strategy
        strategy_counts[strategy] += 1

    total_int4 = strategy_counts['linear'] + strategy_counts['nonlinear']

    print(f"\n   Strategy breakdown:")
    print(f"   • SKIP (FP32/16):  {strategy_counts['skip']} layers")
    print(f"   • FP16:            {strategy_counts['fp16']} layers")
    print(f"   • INT4 (V2):       {total_int4} layers\n")

    compressor = GPTQLiteV2(group_size=128)

    compressed_dict = {}
    fp16_dict = {}
    skip_names = []

    original_size = 0
    compressed_size = 0
    int4_count = 0
    prev_layer_error = None
    start_time = time.time()

    # Calibration texts for activation collection
    calib_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Python is a popular programming language for data science.",
        "Scientists have discovered a new species of dinosaur.",
    ]

    for idx, (name, tensor) in enumerate(state_dict.items()):
        strategy = layer_strategies[name]
        tensor_bytes = tensor.numel() * tensor.element_size()
        original_size += tensor_bytes

        if strategy == 'skip':
            skip_names.append(name)
            compressed_size += tensor_bytes

        elif strategy == 'fp16':
            fp16_dict[name] = tensor.half() if tensor.dtype != torch.float16 else tensor.clone()
            fp16_bytes = tensor.numel() * 2
            compressed_size += fp16_bytes
            if tensor.numel() > 1000:
                print(f"   [FP16] {name} ({list(tensor.shape)}) "
                      f"{tensor_bytes / 1024 / 1024:.1f}MB → {fp16_bytes / 1024 / 1024:.1f}MB")

        else:
            # INT4 compression
            int4_count += 1
            # Convert to FP32 for compression (V2 needs float32)
            weights_np = tensor.detach().cpu().float().numpy()

            elapsed = time.time() - start_time
            print(f"\n   [{int4_count}/{total_int4}] {name}  (elapsed: {elapsed:.0f}s)")
            print(f"      Shape: {list(weights_np.shape)} | Params: {weights_np.size:,}")

            # Collect real activations if possible
            calib_input = None
            if weights_np.ndim == 2:
                try:
                    activations = []

                    def make_hook_fn():
                        def hook_fn(module, inp, out):
                            if isinstance(inp, tuple) and len(inp) > 0:
                                activations.append(inp[0].detach().cpu().float().numpy())
                        return hook_fn

                    # Find the module
                    parent_name = name.replace('.weight', '').replace('.bias', '')
                    for mname, module in model.named_modules():
                        if mname == parent_name:
                            h = module.register_forward_hook(make_hook_fn())
                            break
                    else:
                        h = None

                    if h is not None:
                        for ctext in calib_texts[:2]:
                            enc = tokenizer(ctext, return_tensors="pt", truncation=True, max_length=32)
                            with torch.no_grad():
                                model(**enc)
                        h.remove()

                        if activations:
                            act = np.concatenate(
                                [a.reshape(-1, a.shape[-1]) for a in activations], axis=0
                            )
                            if act.shape[-1] == weights_np.shape[1]:
                                calib_input = act[:32]
                            del activations
                except Exception:
                    pass

            # V2 Compress
            compressed = compressor.compress_v2(
                weights_np,
                layer_name=name,
                layer_idx=int4_count - 1,
                total_layers=total_int4,
                mode=strategy,
                calibration_input=calib_input,
                prev_layer_error=prev_layer_error
            )

            prev_layer_error = compressed.get('layer_error', None)

            # Calculate compressed size
            lcs = compressed['packed'].nbytes
            lcs += compressed['outlier_mask'].nbytes
            lcs += compressed['outlier_values'].nbytes
            for key in ['scales', 'zero_points', 'constants', 'scaling_factor']:
                if compressed.get(key) is not None:
                    lcs += compressed[key].nbytes

            compressed_size += lcs

            # Store compressed (without layer_error to save memory)
            comp_store = {k: v for k, v in compressed.items() if k != 'layer_error'}
            compressed_dict[name] = comp_store

            ratio = weights_np.nbytes / lcs if lcs > 0 else 0
            print(f"      {weights_np.nbytes / 1024 / 1024:.1f}MB → {lcs / 1024 / 1024:.1f}MB ({ratio:.2f}x)")

            # Free memory
            del weights_np, calib_input
            gc.collect()

            # Progress report every 10 layers
            if int4_count % 10 == 0:
                print_ram()

    compress_time = time.time() - start_time

    print(f"\n   ⏱️  Compression completed in {compress_time:.0f}s")
    print_ram()

    # ================================================================
    # Step 4: Rebuild model
    # ================================================================
    print("\n" + "=" * 70)
    print("🔧 REBUILDING COMPRESSED MODEL...")
    print("=" * 70)

    rebuild_start = time.time()
    new_state_dict = {}

    for name, tensor in state_dict.items():
        strategy = layer_strategies[name]

        if strategy == 'skip':
            new_state_dict[name] = tensor.clone()
        elif strategy == 'fp16':
            # Keep as FP16 (model is already FP16)
            new_state_dict[name] = fp16_dict[name]
        else:
            restored_np = compressor.decompress_v2(compressed_dict[name])
            new_state_dict[name] = torch.from_numpy(restored_np).to(tensor.dtype)

    # Free original state dict
    del state_dict
    gc.collect()

    # Load into fresh model
    print("   Loading fresh model shell...")
    try:
        model_comp = AutoModelForCausalLM.from_pretrained(
            actual_model_id,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    except Exception:
        model_comp = AutoModelForCausalLM.from_pretrained(
            actual_model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
    model_comp.load_state_dict(new_state_dict, strict=False)
    model_comp.eval()

    rebuild_time = time.time() - rebuild_start
    print(f"   ✅ Rebuilt in {rebuild_time:.1f}s")

    # Free original model
    del model
    gc.collect()
    print_ram()

    # ================================================================
    # Step 5: Test compressed model
    # ================================================================
    print("\n" + "─" * 70)
    print("📝 COMPRESSED MODEL: Generating text...")
    print("─" * 70)

    print("   ⏳ Generating (CPU)...")
    for prompt in test_prompts:
        try:
            text = generate_text_cpu(model_comp, tokenizer, prompt, max_new_tokens=40)
            print(f"\n  Prompt: '{prompt}'")
            print(f"  Output: {text[len(prompt):][:200]}")
        except Exception as e:
            print(f"  ⚠️ Generation failed: {e}")

    # ================================================================
    # Step 6: Quality metrics
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 QUALITY COMPARISON")
    print("=" * 70)

    try:
        comp_ppl = compute_perplexity_cpu(model_comp, tokenizer, eval_text)
        print(f"\n  Perplexity:")
        print(f"    Original:   {orig_ppl:.2f}" if orig_ppl else "    Original:   N/A")
        print(f"    Compressed: {comp_ppl:.2f}")

        if orig_ppl:
            ppl_increase = ((comp_ppl - orig_ppl) / orig_ppl) * 100
            print(f"    Increase:   {ppl_increase:+.1f}%")

            if ppl_increase < 5:
                ppl_v = "✅✅ EXCELLENT (< 5%)"
            elif ppl_increase < 15:
                ppl_v = "✅ GOOD (< 15%)"
            elif ppl_increase < 50:
                ppl_v = "⚠️ ACCEPTABLE (< 50%)"
            else:
                ppl_v = "❌ NEEDS IMPROVEMENT"
            print(f"    Verdict:    {ppl_v}")
        else:
            ppl_increase = None
            ppl_v = "N/A"
    except Exception as e:
        comp_ppl = None
        ppl_increase = None
        ppl_v = f"Error: {e}"
        print(f"  ⚠️ PPL computation failed: {e}")

    # ================================================================
    # Step 7: Save
    # ================================================================
    model_short = actual_model_id.split('/')[-1].lower().replace('-', '_')
    save_path = f"{model_short}_v2_compressed.agi"
    print(f"\n💾 Saving to {save_path}...")

    save_data = {
        'compressed_layers': {},
        'fp16_layers': fp16_dict,
        'skip_layer_names': skip_names,
        'model_id': actual_model_id,
        'compressor': 'GPTQ-Lite-V2',
        'metrics': {
            'original_ppl': orig_ppl,
            'compressed_ppl': comp_ppl,
            'ppl_increase': ppl_increase,
            'compression_ratio': original_size / compressed_size if compressed_size > 0 else 0,
        }
    }

    for k, v in compressed_dict.items():
        save_data['compressed_layers'][k] = {
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
        }

    torch.save(save_data, save_path)
    file_size = os.path.getsize(save_path)

    # ================================================================
    # Final Report
    # ================================================================
    print("\n" + "#" * 70)
    print("  📋 FINAL REPORT — LLAMA V2 COMPRESSION")
    print("#" * 70)
    print(f"  Model:              {actual_model_id}")
    print(f"  Parameters:         {total_params:,}")
    print(f"  Method:             GPTQ-Lite V2 (Refined)")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Strategy:")
    print(f"    FP32/16 (skip):   {strategy_counts['skip']} layers")
    print(f"    FP16:             {strategy_counts['fp16']} layers")
    print(f"    INT4 (V2):        {total_int4} layers")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Original Size:      {original_size / 1024**3:.2f} GB")
    print(f"  Compressed Size:    {compressed_size / 1024**3:.2f} GB")
    print(f"  File on Disk:       {file_size / 1024**3:.2f} GB")
    print(f"  Compression:        {original_size / compressed_size:.2f}x")
    print(f"  ─────────────────────────────────────────────")
    if orig_ppl:
        print(f"  Original PPL:       {orig_ppl:.2f}")
    if comp_ppl:
        print(f"  Compressed PPL:     {comp_ppl:.2f}")
    if ppl_increase is not None:
        print(f"  PPL Increase:       {ppl_increase:+.1f}%")
    print(f"  ─────────────────────────────────────────────")
    print(f"  Compress Time:      {compress_time:.0f}s ({compress_time/60:.1f} min)")
    print(f"  Quality:            {ppl_v}")
    print("#" * 70)

    return save_data


if __name__ == "__main__":
    model_id = None
    if len(sys.argv) > 1:
        model_id = sys.argv[1].strip()

    print("  Usage: python Compress_Llama3_V2.py [model_id]")
    print("  Examples:")
    print("    python Compress_Llama3_V2.py meta-llama/Meta-Llama-3-8B")
    print("    python Compress_Llama3_V2.py meta-llama/Llama-3.2-3B")
    print("    python Compress_Llama3_V2.py microsoft/phi-2")
    print("    python Compress_Llama3_V2.py  (auto-detect)\n")

    compress_llama3(model_id)
