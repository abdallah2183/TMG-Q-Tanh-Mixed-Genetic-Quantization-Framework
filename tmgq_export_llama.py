import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tmgq_packer import QuantizedLinear, pack_3bit
import math

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    import functools
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def sensitivity_quantize_packable(w, n_bits, gs=128):
    """Directly extracts pure INT limits and FP16 scales without reconstruct decay."""
    q_levels = (2**n_bits) - 1
    rows, cols = w.shape
    num_blocks = math.ceil(cols / gs)
    
    limits_int = torch.zeros_like(w, dtype=torch.int32)
    scales = torch.zeros((rows, num_blocks), dtype=torch.float16, device=w.device)
    zeros = torch.zeros((rows, num_blocks), dtype=torch.float16, device=w.device)
    
    for i, cs in enumerate(range(0, cols, gs)):
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
        
        # Robust Int Mapping (Eliminates float16 Inf Overflow)
        ws = (block - b_min) / scale
        wr = torch.clamp(torch.round(ws), 0, q_levels).to(torch.int32)
        
        limits_int[:, cs:ce] = wr
        scales[:, i:i+1] = scale.to(torch.float16)
        zeros[:, i:i+1] = b_min.to(torch.float16)
        
    return limits_int, scales, zeros

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
             
        print(f"  [{current_layer}/{total_layers}] Extracting pure INT bounds & Packing: {n} ({w.shape[0]}x{w.shape[1]})", end='\r')
        
        limits_int, scales, zeros = sensitivity_quantize_packable(w, bits, gs=128)
        
        # Deploy the TMG-Q Packer manually without float drift
        qlayer = QuantizedLinear(m.in_features, m.out_features, bias=m.bias is not None, gs=128)
        packed_w, shape, pad = pack_3bit(limits_int)
        
        qlayer.qweight = packed_w
        qlayer.scales = scales
        qlayer.zeros = zeros
        qlayer.w_shape = torch.tensor(shape, dtype=torch.int32)
        qlayer.pad_len = torch.tensor(pad, dtype=torch.int32)
        if m.bias is not None:
             qlayer.bias.data = m.bias.data.to(torch.float16)
        
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
