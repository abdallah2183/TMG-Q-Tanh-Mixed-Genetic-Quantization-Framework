import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tmgq_packer import QuantizedLinear
import math

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    import functools
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def sensitivity_quantize(w, n_bits, h_diag=None, gs=128):
    q_levels = (2**n_bits) - 1
    rows, cols = w.shape
    w_q = torch.zeros_like(w)
    for cs in range(0, cols, gs):
        ce = min(cs+gs, cols)
        block = w[:, cs:ce].clone()
        med = block.median()
        mad = (block - med).abs().median()
        sigma = 1.4826 * mad
        clamp_min = med - (3.5 * sigma)
        clamp_max = med + (3.5 * sigma)
        block = torch.clamp(block, clamp_min, clamp_max)
        b_min = block.min(dim=1, keepdim=True).values
        b_max = block.max(dim=1, keepdim=True).values
        scale = (b_max - b_min).clamp(min=1e-8) / q_levels
        zero_point = torch.round(-b_min / scale)
        ws = (block / scale) + zero_point
        wr = torch.clamp(torch.round(ws), 0, q_levels)
        w_q[:, cs:ce] = (wr - zero_point) * scale
    return w_q

def error_diffusion(orig, wc, h_diag=None, n_waves=3, gs=128):
    res = orig - wc
    for wave in range(n_waves):
        wave_scale = 1.0 / (2 ** wave)
        res_scaled = res * wave_scale
        rqmax = res_scaled.max()
        rqmin = res_scaled.min()
        if rqmax == rqmin:
            break
        for cs in range(0, res.shape[1], gs):
            ce = min(cs+gs, res.shape[1])
            blk = res_scaled[:, cs:ce]
            sc = blk.abs().max(1, keepdim=True).values.clamp(min=1e-8)/rqmax
            res_scaled[:, cs:ce] = torch.clamp(torch.round(blk/sc), rqmin, rqmax)*sc
        wc = wc + res_scaled
        res = orig - wc
    return wc

def export_huggingface_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0", bits=3, export_path="TinyLlama_3bit_TMGQ.pt"):
    print(f"Loading base FP16 model: {model_name} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    
    total_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    current_layer = 1
    
    print("\nStarting TMG-Q Ultra Advanced Packing Sequence...")
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    
    for n in linear_names:
        m = rgetattr(model, n)
        w = m.weight.data
        if "lm_head" in n:
             # Skip output embeddings to preserve reasoning accuracy
             current_layer += 1
             continue
             
        print(f"  [{current_layer}/{total_layers}] Compressing & Packing: {n} ({w.shape[0]}x{w.shape[1]})", end='\r')
        orig = w.clone()
        
        wq = sensitivity_quantize(w, bits)
        wq = error_diffusion(orig, wq)
        
        # Deploy the TMG-Q Packer!
        qlayer = QuantizedLinear(m.in_features, m.out_features, bias=m.bias is not None, gs=128)
        qlayer.pack_from_float(wq, m.bias.data if m.bias is not None else None, n_bits=bits)
        
        pre, _, post = n.rpartition('.')
        parent = rgetattr(model, pre) if pre else model
        setattr(parent, post, qlayer)
        current_layer += 1
        
    print(f"\nTMG-Q Packing Complete. Extracting state_dict...")
    
    # Save the completely shrunken architecture
    torch.save(model.state_dict(), export_path)
    print(f"\n=========================================")
    print(f"✅ EXPORT SUCCESSFUL!")
    print(f"Compressed Matrix saved to: {export_path}")
    print(f"You can now send {export_path} to your friend!")
    print(f"=========================================")

if __name__ == "__main__":
    export_huggingface_model()
