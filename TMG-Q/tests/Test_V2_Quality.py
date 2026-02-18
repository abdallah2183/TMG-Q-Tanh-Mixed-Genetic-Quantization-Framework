"""
=================================================================
TMG-Q Compression Quality Test on LLaVA 7B
  Tanh-Mixed-Genetic Quantization Framework
=================================================================
اختبار شامل:
  1. مقارنة أوزان TMG-Q vs Q4_0 (MSE, Cosine Similarity, Max Error)
  2. Forward pass محاكاة: مقارنة المخرجات
  3. اختبار Ollama المباشر
  4. تحليل التوزيع الإحصائي
"""

import numpy as np
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from TMG_Q import TMGQ


# ================================================================
# GGUF Dequantizer (Vectorized)
# ================================================================
def dequantize_q4_0(data_bytes, shape):
    """Vectorized Q4_0 dequantization."""
    shape = [int(s) for s in shape]
    total = 1
    for s in shape:
        total *= s
    
    n_blocks = total // 32
    data = np.frombuffer(data_bytes, dtype=np.uint8)[:n_blocks * 18]
    data = data.reshape(n_blocks, 18)
    
    scales = data[:, :2].copy().view(np.float16).astype(np.float32).flatten()
    quant = data[:, 2:18]
    
    low = (quant & 0x0F).astype(np.int8) - 8
    high = ((quant >> 4) & 0x0F).astype(np.int8) - 8
    
    result = np.empty((n_blocks, 32), dtype=np.float32)
    result[:, 0::2] = low.astype(np.float32)
    result[:, 1::2] = high.astype(np.float32)
    result *= scales[:, np.newaxis]
    
    return result.reshape(shape)


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = np.dot(a_flat, b_flat)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(dot / (norm_a * norm_b))


def test_v2_quality():
    """Comprehensive V2 quality test on LLaVA 7B."""
    
    gguf_path = r"C:\Users\abdal\.ollama\models\blobs\sha256-170370233dd5c5415250a2ecd5c71586352850729062ccef1496385647293868"
    
    print("\n" + "=" * 70)
    print("🧪 TMG-Q COMPRESSION QUALITY TEST — LLaVA 7B")
    print("   Tanh-Mixed-Genetic Quantization Framework")
    print("=" * 70)
    
    from gguf import GGUFReader
    reader = GGUFReader(gguf_path)
    print(f"\n📥 Loaded {len(reader.tensors)} tensors from GGUF")
    
    print("\n" + "=" * 70)
    print("📊 TEST 1: Per-Layer Quality Analysis")
    print("   Dequantize Q4_0 → Compress TMG-Q → Decompress → Compare")
    print("=" * 70)
    
    compressor = TMGQ(group_size=128)
    
    # Select diverse layers to test
    test_layers = {
        # Early layers (foundation)
        'blk.2.attn_q.weight': 'Attention Q (early)',
        'blk.2.ffn_gate.weight': 'MLP Gate (early)',
        # Middle layers (processing)
        'blk.15.attn_q.weight': 'Attention Q (middle)',
        'blk.15.ffn_down.weight': 'MLP Down (middle)',
        'blk.15.attn_v.weight': 'Attention V (middle)',
        # Late layers (output)
        'blk.28.attn_q.weight': 'Attention Q (late)',
        'blk.28.ffn_gate.weight': 'MLP Gate (late)',
        'blk.28.attn_output.weight': 'Attention Out (late)',
    }
    
    results = []
    layer_idx = 0
    total_test = len(test_layers)
    
    for tensor_info in reader.tensors:
        if tensor_info.name not in test_layers:
            continue
        
        layer_idx += 1
        name = tensor_info.name
        label = test_layers[name]
        shape = [int(s) for s in tensor_info.shape]
        total_elements = 1
        for s in shape:
            total_elements *= s
        
        print(f"\n  [{layer_idx}/{total_test}] {name}")
        print(f"  Label: {label}")
        print(f"  Shape: {shape} | Elements: {total_elements:,}")
        
        # Step 1: Dequantize from Q4_0
        tensor_data = bytes(tensor_info.data)
        original_fp32 = dequantize_q4_0(tensor_data, shape)
        
        # Step 2: Compress with TMG-Q
        calib_input = np.random.randn(32, original_fp32.shape[1]).astype(np.float32) * 0.1
        
        compressed = compressor.compress(
            original_fp32,
            layer_name=name,
            layer_idx=layer_idx,
            total_layers=total_test,
            mode='linear',
            calibration_input=calib_input,
            prev_layer_error=None
        )
        
        # Step 3: Decompress
        restored = compressor.decompress(compressed)
        
        # Step 4: Quality metrics
        error = original_fp32 - restored
        mse = float(np.mean(error ** 2))
        mae = float(np.mean(np.abs(error)))
        max_err = float(np.max(np.abs(error)))
        cos_sim = cosine_similarity(original_fp32, restored)
        
        # Signal-to-Noise Ratio
        signal_power = float(np.mean(original_fp32 ** 2))
        snr = 10 * np.log10(signal_power / mse) if mse > 0 else float('inf')
        
        # Weight distribution stats
        orig_mean = float(np.mean(original_fp32))
        orig_std = float(np.std(original_fp32))
        rest_mean = float(np.mean(restored))
        rest_std = float(np.std(restored))
        
        # Percentage of weights with error > 1%
        relative_error = np.abs(error) / (np.abs(original_fp32) + 1e-10)
        pct_above_1 = float(np.mean(relative_error > 0.01) * 100)
        pct_above_5 = float(np.mean(relative_error > 0.05) * 100)
        pct_above_10 = float(np.mean(relative_error > 0.10) * 100)
        
        # Compression ratio
        q4_size = (total_elements // 32) * 18
        v2_size = compressed['packed'].nbytes + compressed['outlier_mask'].nbytes + compressed['outlier_values'].nbytes
        for key in ['scales', 'zero_points', 'constants', 'scaling_factor']:
            if compressed.get(key) is not None:
                v2_size += compressed[key].nbytes
        
        result = {
            'name': name, 'label': label,
            'mse': mse, 'mae': mae, 'max_err': max_err,
            'cos_sim': cos_sim, 'snr': snr,
            'q4_ratio': (total_elements * 4) / q4_size,
            'v2_ratio': (total_elements * 4) / v2_size,
            'pct_above_1': pct_above_1,
            'pct_above_5': pct_above_5,
            'pct_above_10': pct_above_10,
            'outlier_pct': compressed['outlier_percent'],
            'scaling_factor': float(compressed['scaling_factor'][0]),
        }
        results.append(result)
        
        print(f"  ┌─────────────────────────────────────────────────")
        print(f"  │ MSE:              {mse:.8f}")
        print(f"  │ MAE:              {mae:.8f}")
        print(f"  │ Max Error:        {max_err:.6f}")
        print(f"  │ Cosine Sim:       {cos_sim:.8f}")
        print(f"  │ SNR:              {snr:.1f} dB")
        print(f"  │ Mean (orig→rest): {orig_mean:.6f} → {rest_mean:.6f}")
        print(f"  │ Std  (orig→rest): {orig_std:.6f} → {rest_std:.6f}")
        print(f"  │ Outlier %:        {compressed['outlier_percent']:.1f}%")
        print(f"  │ Scale Factor:     {compressed['scaling_factor'][0]:.4f}")
        print(f"  │ Error > 1%:       {pct_above_1:.1f}% of weights")
        print(f"  │ Error > 5%:       {pct_above_5:.1f}% of weights")
        print(f"  │ Error > 10%:      {pct_above_10:.1f}% of weights")
        print(f"  │ Q4_0 ratio:       {result['q4_ratio']:.1f}x")
        print(f"  │ TMG-Q ratio:      {result['v2_ratio']:.1f}x")
        print(f"  └─────────────────────────────────────────────────")
        
        del original_fp32, restored, compressed
    
    # ================================================================
    # TEST 2: Simulated Forward Pass
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 TEST 2: Simulated Forward Pass")
    print("   Feed same input through original vs TMG-Q-restored weights")
    print("=" * 70)
    
    # Pick a few sequential layers for forward pass simulation
    forward_layers = ['blk.15.attn_q.weight', 'blk.15.attn_k.weight',
                      'blk.15.attn_v.weight', 'blk.15.attn_output.weight']
    
    print(f"\n  Simulating forward pass with matching input dimensions...")
    
    for tensor_info in reader.tensors:
        if tensor_info.name not in forward_layers:
            continue
        
        name = tensor_info.name
        shape = [int(s) for s in tensor_info.shape]
        
        # Create input matching this weight's input dimension
        input_dim = shape[1]  # W is [out, in], so input must be [batch, in]
        sim_input = np.random.randn(4, input_dim).astype(np.float32) * 0.02
        
        # Dequantize
        tensor_data = bytes(tensor_info.data)
        orig_w = dequantize_q4_0(tensor_data, shape)
        
        # TMG-Q compress → decompress
        compressed = compressor.compress(
            orig_w, layer_name=name, layer_idx=0, total_layers=4,
            mode='linear', calibration_input=None, prev_layer_error=None
        )
        rest_w = compressor.decompress(compressed)
        
        # Forward pass
        out_orig = sim_input @ orig_w.T
        out_rest = sim_input @ rest_w.T
        
        # Output comparison
        out_mse = float(np.mean((out_orig - out_rest) ** 2))
        out_cos = cosine_similarity(out_orig, out_rest)
        out_max = float(np.max(np.abs(out_orig - out_rest)))
        
        print(f"\n  Layer: {name}")
        print(f"    Weight shape:    {shape}")
        print(f"    Output shape:    {list(out_orig.shape)}")
        print(f"    Output MSE:      {out_mse:.8f}")
        print(f"    Output Cosine:   {out_cos:.8f}")
        print(f"    Output Max Err:  {out_max:.6f}")
        
        if out_cos > 0.999:
            print(f"    Verdict:         ✅✅ EXCELLENT (cos > 0.999)")
        elif out_cos > 0.99:
            print(f"    Verdict:         ✅ GOOD (cos > 0.99)")
        elif out_cos > 0.95:
            print(f"    Verdict:         ⚠️ ACCEPTABLE (cos > 0.95)")
        else:
            print(f"    Verdict:         ❌ NEEDS IMPROVEMENT")
        
        del orig_w, rest_w, compressed
    
    # ================================================================
    # TEST 3: Ollama Live Test
    # ================================================================
    print("\n" + "=" * 70)
    print("📊 TEST 3: Ollama Live Generation Test")
    print("=" * 70)
    
    test_prompts = [
        "The future of artificial intelligence is",
        "Explain quantum computing in simple terms:",
        "Write a Python function to sort a list:",
        "What is the difference between machine learning and deep learning?",
        "Translate to Arabic: Hello, how are you today?",
    ]
    
    try:
        import urllib.request
        import json
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n  [{i}/{len(test_prompts)}] Prompt: '{prompt}'")
            
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps({
                    "model": "llava",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"num_predict": 80, "temperature": 0.7, "seed": 42}
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            
            try:
                start = time.time()
                with urllib.request.urlopen(req, timeout=120) as resp:
                    result = json.loads(resp.read())
                elapsed = time.time() - start
                
                response = result.get('response', '')
                tokens = result.get('eval_count', 0)
                tok_per_sec = tokens / elapsed if elapsed > 0 else 0
                
                print(f"  Response ({elapsed:.1f}s, {tok_per_sec:.1f} tok/s):")
                print(f"  {response[:300]}")
                
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
                
    except Exception as e:
        print(f"  ⚠️ Ollama not available: {e}")
    
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "#" * 70)
    print("  📋 TMG-Q QUALITY TEST SUMMARY")
    print("#" * 70)
    
    if results:
        avg_mse = np.mean([r['mse'] for r in results])
        avg_mae = np.mean([r['mae'] for r in results])
        avg_cos = np.mean([r['cos_sim'] for r in results])
        avg_snr = np.mean([r['snr'] for r in results if r['snr'] < float('inf')])
        avg_v2 = np.mean([r['v2_ratio'] for r in results])
        avg_q4 = np.mean([r['q4_ratio'] for r in results])
        avg_pct1 = np.mean([r['pct_above_1'] for r in results])
        avg_pct5 = np.mean([r['pct_above_5'] for r in results])
        avg_pct10 = np.mean([r['pct_above_10'] for r in results])
        
        print(f"\n  Test layers: {len(results)}")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Average MSE:          {avg_mse:.8f}")
        print(f"  Average MAE:          {avg_mae:.8f}")
        print(f"  Average Cosine Sim:   {avg_cos:.8f}")
        print(f"  Average SNR:          {avg_snr:.1f} dB")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Weights with >1% err: {avg_pct1:.1f}%")
        print(f"  Weights with >5% err: {avg_pct5:.1f}%")
        print(f"  Weights with >10% err:{avg_pct10:.1f}%")
        print(f"  ─────────────────────────────────────────────")
        print(f"  Avg Q4_0 ratio:       {avg_q4:.1f}x")
        print(f"  Avg TMG-Q ratio:      {avg_v2:.1f}x")
        print(f"  TMG-Q / Q4_0 effic:   {avg_v2 / avg_q4 * 100:.0f}%")
        print(f"  ─────────────────────────────────────────────")
        
        # Overall verdict
        if avg_cos > 0.9999:
            verdict = "✅✅✅ OUTSTANDING — Near-lossless quality"
        elif avg_cos > 0.999:
            verdict = "✅✅ EXCELLENT — Production-ready quality"
        elif avg_cos > 0.99:
            verdict = "✅ GOOD — Acceptable for most use cases"
        elif avg_cos > 0.95:
            verdict = "⚠️ ACCEPTABLE — Some quality loss"
        else:
            verdict = "❌ NEEDS IMPROVEMENT"
        
        print(f"\n  🏆 OVERALL VERDICT: {verdict}")
    
    print("#" * 70)


if __name__ == "__main__":
    test_v2_quality()
