"""
=================================================================
💬 GPT-2 Compressed Chat — TMG-Q
    Tanh-Mixed-Genetic Quantization Framework
=================================================================
1. تحميل GPT-2 Medium (355M parameters)
2. ضغط بـ TMG-Q مع Mixed Precision
3. مقارنة الحجم الأصلي vs المضغوط
4. محادثة تفاعلية مباشرة — اكتب واحكي مع النموذج!
"""

import torch
import numpy as np
import sys
import os
import time
import gc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Suppress warnings
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'


def get_model_size_mb(model):
    """Get model size in MB."""
    total = sum(p.nelement() * p.element_size() for p in model.parameters())
    return total / (1024 * 1024)


def classify_gpt2_layer(name, num_blocks=24):
    """Smart layer classification for GPT-2."""
    n = name.lower()
    
    # NEVER compress embeddings and norms
    if any(k in n for k in ['wte', 'wpe', 'embed']):
        return 'skip'
    if any(k in n for k in ['ln_', 'layernorm', 'layer_norm', 'ln_f']):
        return 'skip'
    # Biases are small, skip
    if n.endswith('.bias'):
        return 'skip'
    
    # lm_head → FP16
    if 'lm_head' in n:
        return 'fp16'
    
    # Extract block number
    import re
    m = re.search(r'h\.(\d+)\.', n)
    if m:
        blk = int(m.group(1))
        # First 2 and last 2 blocks → FP16
        if blk <= 1 or blk >= (num_blocks - 2):
            return 'fp16'
    
    # Everything else → INT4 V2
    return 'linear'


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from TMG_Q import TMGQ
    
    print("\n" + "=" * 70)
    print("💬 GPT-2 COMPRESSED CHAT — TMG-Q")
    print("   Tanh-Mixed-Genetic Quantization Framework")
    print("=" * 70)
    
    # ================================================================
    # Step 1: Load Model
    # ================================================================
    model_id = "gpt2-medium"  # 355M params — good quality
    
    print(f"\n📥 Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32
    )
    model.eval()
    
    original_size_mb = get_model_size_mb(model)
    total_params = sum(p.numel() for p in model.parameters())
    
    # Count blocks
    num_blocks = len([n for n, _ in model.named_parameters() if '.h.' in n and '.attn.c_attn.weight' in n])
    
    print(f"   ✅ Loaded! {total_params/1e6:.0f}M parameters")
    print(f"   📦 Original size: {original_size_mb:.0f} MB (FP32)")
    print(f"   🏗️  Blocks: {num_blocks}")
    
    # ================================================================
    # Step 2: Baseline test
    # ================================================================
    print("\n" + "─" * 70)
    print("📝 BASELINE (Before Compression)")
    print("─" * 70)
    
    test_prompts = [
        "The meaning of life is",
        "Artificial intelligence will",
        "The best programming language is",
    ]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, temperature=0.7,
                do_sample=True, top_p=0.9, top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n  > {prompt}")
        print(f"  < {text[:200]}")
    
    # Save baseline perplexity
    ppl_text = "The quick brown fox jumps over the lazy dog. Machine learning is a subset of artificial intelligence."
    ppl_input = tokenizer(ppl_text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**ppl_input, labels=ppl_input["input_ids"])
        baseline_ppl = torch.exp(outputs.loss).item()
    print(f"\n  📊 Baseline Perplexity: {baseline_ppl:.2f}")
    
    # ================================================================
    # Step 3: Compress with V2
    # ================================================================
    print("\n" + "=" * 70)
    print("⚙️  COMPRESSING WITH TMG-Q")
    print("   Strategy: Skip embeddings/norms, FP16 first/last blocks, INT4 middle")
    print("=" * 70)
    
    compressor = TMGQ(group_size=128)
    state_dict = model.state_dict()
    
    compressed_data = {}
    new_state_dict = {}
    
    stats = {'skip': 0, 'fp16': 0, 'int4': 0}
    original_bytes = 0
    compressed_bytes = 0
    prev_error = None
    int4_idx = 0
    
    # Count INT4 layers for progress
    total_int4 = sum(1 for n in state_dict if classify_gpt2_layer(n, num_blocks) == 'linear' 
                     and state_dict[n].dim() >= 2 and state_dict[n].numel() >= 1024)
    
    start_time = time.time()
    
    for name, param in state_dict.items():
        tensor = param.numpy() if param.dtype == torch.float32 else param.float().numpy()
        strategy = classify_gpt2_layer(name, num_blocks)
        param_bytes = tensor.nbytes
        original_bytes += param_bytes
        
        if strategy == 'skip':
            new_state_dict[name] = param
            compressed_bytes += param_bytes
            stats['skip'] += 1
            continue
        
        if strategy == 'fp16':
            fp16_tensor = param.half()
            new_state_dict[name] = fp16_tensor.float()
            compressed_bytes += fp16_tensor.nelement() * 2
            stats['fp16'] += 1
            continue
        
        # INT4 V2
        if tensor.ndim < 2 or tensor.size < 1024:
            new_state_dict[name] = param
            compressed_bytes += param_bytes
            stats['skip'] += 1
            continue
        
        int4_idx += 1
        print(f"   [{int4_idx}/{total_int4}] {name} {list(tensor.shape)}")
        
        # Calibration
        calib = np.random.randn(32, tensor.shape[1]).astype(np.float32) * 0.1
        
        # Compress
        result = compressor.compress(
            tensor, layer_name=name, layer_idx=int4_idx,
            total_layers=total_int4, mode='linear',
            calibration_input=calib, prev_layer_error=prev_error
        )
        prev_error = result.get('layer_error')
        
        # Decompress immediately (we need the model for inference)
        restored = compressor.decompress(result)
        new_state_dict[name] = torch.from_numpy(restored).float()
        
        # Size tracking
        layer_compressed = (result['packed'].nbytes + result['outlier_mask'].nbytes + 
                          result['outlier_values'].nbytes)
        for key in ['scales', 'zero_points', 'constants', 'scaling_factor']:
            if result.get(key) is not None:
                layer_compressed += result[key].nbytes
        compressed_bytes += layer_compressed
        
        compressed_data[name] = result
        stats['int4'] += 1
        
        del result, restored
    
    compress_time = time.time() - start_time
    
    # ================================================================
    # Step 4: Rebuild model
    # ================================================================
    print("\n" + "─" * 70)
    print("🔧 REBUILDING COMPRESSED MODEL...")
    print("─" * 70)
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    
    compressed_size_mb = compressed_bytes / (1024 * 1024)
    ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 0
    savings = (1 - compressed_bytes / original_bytes) * 100
    
    print(f"\n  📊 COMPRESSION RESULTS:")
    print(f"  ┌──────────────────────────────────────────")
    print(f"  │ Original:      {original_size_mb:.0f} MB (FP32)")
    print(f"  │ Compressed:    {compressed_size_mb:.0f} MB")
    print(f"  │ Ratio:         {ratio:.2f}x")
    print(f"  │ Savings:       {savings:.1f}%")
    print(f"  │ Time:          {compress_time:.0f}s")
    print(f"  │ ────────────────────────────────────────")
    print(f"  │ Layers skipped:   {stats['skip']}")
    print(f"  │ Layers FP16:      {stats['fp16']}")
    print(f"  │ Layers INT4 TMG-Q: {stats['int4']}")
    print(f"  └──────────────────────────────────────────")
    
    # ================================================================
    # Step 5: Test compressed model
    # ================================================================
    print("\n" + "─" * 70)
    print("📝 AFTER COMPRESSION TEST")
    print("─" * 70)
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=50, temperature=0.7,
                do_sample=True, top_p=0.9, top_k=50,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        print(f"\n  > {prompt}")
        print(f"  < {text[:200]}")
    
    # Compressed perplexity
    with torch.no_grad():
        outputs = model(**ppl_input, labels=ppl_input["input_ids"])
        compressed_ppl = torch.exp(outputs.loss).item()
    
    ppl_change = ((compressed_ppl - baseline_ppl) / baseline_ppl) * 100
    
    print(f"\n  📊 Compressed Perplexity: {compressed_ppl:.2f}")
    print(f"  📊 Baseline Perplexity:  {baseline_ppl:.2f}")
    print(f"  📊 Change:               {ppl_change:+.1f}%")
    
    if abs(ppl_change) < 10:
        print("  ✅ EXCELLENT — Quality preserved!")
    elif abs(ppl_change) < 25:
        print("  ✅ GOOD — Minor quality impact")
    else:
        print("  ⚠️ NOTICEABLE — Some quality degradation")
    
    # ================================================================
    # Step 6: Interactive Chat!
    # ================================================================
    print("\n" + "=" * 70)
    print("💬 INTERACTIVE CHAT — TMG-Q Compressed GPT-2")
    print("   Tanh-Mixed-Genetic Quantization Framework")
    print("=" * 70)
    print(f"  Model: {model_id} — Compressed {ratio:.1f}x with TMG-Q")
    print(f"  Type your message and press Enter")
    print(f"  Commands: /quit (exit) | /temp 0.8 (set temperature)")
    print(f"            /tokens 100 (max tokens) | /reset (clear history)")
    print("=" * 70)
    
    temperature = 0.7
    max_tokens = 80
    chat_history = ""
    
    while True:
        try:
            user_input = input("\n  🧑 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        # Commands
        if user_input.startswith('/'):
            cmd = user_input.lower().split()
            if cmd[0] == '/quit' or cmd[0] == '/exit':
                print("  👋 Goodbye!")
                break
            elif cmd[0] == '/temp' and len(cmd) > 1:
                try:
                    temperature = float(cmd[1])
                    print(f"  ⚙️  Temperature set to {temperature}")
                except:
                    print("  ⚠️ Usage: /temp 0.8")
                continue
            elif cmd[0] == '/tokens' and len(cmd) > 1:
                try:
                    max_tokens = int(cmd[1])
                    print(f"  ⚙️  Max tokens set to {max_tokens}")
                except:
                    print("  ⚠️ Usage: /tokens 100")
                continue
            elif cmd[0] == '/reset':
                chat_history = ""
                print("  🔄 Chat history cleared!")
                continue
            elif cmd[0] == '/stats':
                print(f"  📊 Model: {model_id} (TMG-Q Compressed)")
                print(f"  📊 TMG-Q Compression: {ratio:.2f}x ({savings:.1f}% smaller)")
                print(f"  📊 Perplexity: {compressed_ppl:.2f} (baseline: {baseline_ppl:.2f})")
                print(f"  📊 Temperature: {temperature}")
                print(f"  📊 Max tokens: {max_tokens}")
                continue
            elif cmd[0] == '/help':
                print("  /quit    — Exit chat")
                print("  /temp N  — Set temperature (0.1-2.0)")
                print("  /tokens N — Set max response tokens")
                print("  /reset   — Clear conversation context")
                print("  /stats   — Show model stats")
                continue
        
        # Build prompt with context
        if chat_history:
            prompt = chat_history + "\n" + user_input
        else:
            prompt = user_input
        
        # Keep context reasonable (last 500 chars)
        if len(prompt) > 500:
            prompt = prompt[-500:]
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        
        start = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),
                do_sample=True,
                top_p=0.9,
                top_k=50,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
            )
        elapsed = time.time() - start
        
        # Extract only the new generated text
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        response = full_text[len(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True)):].strip()
        
        if not response:
            response = full_text.strip()
        
        # Clean up response
        # Cut at natural stopping point
        for stop in ['\n\n\n', '.\n\n']:
            if stop in response:
                response = response[:response.index(stop) + 1]
                break
        
        tokens_generated = out.shape[1] - inputs['input_ids'].shape[1]
        tok_per_sec = tokens_generated / elapsed if elapsed > 0 else 0
        
        print(f"  🤖 GPT-2: {response}")
        print(f"     ({tokens_generated} tokens, {elapsed:.1f}s, {tok_per_sec:.0f} tok/s)")
        
        # Update history
        chat_history = prompt + " " + response


if __name__ == "__main__":
    main()
