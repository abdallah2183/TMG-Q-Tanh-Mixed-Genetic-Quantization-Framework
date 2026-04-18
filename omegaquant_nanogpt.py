#!/usr/bin/env python
"""
TMG-Q Ultra CLI (nanoGPT Custom Interface)
==============================================
Phase 3 Edition: Asymmetric Scale, 3-Sigma Outlier Shield, NO SVD LEAKAGE.
Professional Command Line Interface for extreme local LLM compression.

Usage:
    python omegaquant_nanogpt.py --in-ckpt out-self-code/ckpt.pt --out-ckpt out-self-code/ckpt_3bit.pt --bits 3
"""
import sys
import copy
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pickle

from tmgq_packer import QuantizedLinear

# Ensure nanoGPT is accessible
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
DATASET_DIR = ROOT / "data" / "self_code_char"
sys.path.insert(0, str(DATASET_DIR))

try:
    from model import GPT, GPTConfig
    from curriculum import build_examples, build_prompt
except ImportError:
    print("Error: Could not import nanoGPT model or curriculum. Make sure you are in the nanoGPT directory.")
    sys.exit(1)


def load_model_from_checkpoint(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)
    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, checkpoint


def load_meta():
    try:
        with open(DATASET_DIR / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        return meta["stoi"], meta["itos"]
    except Exception as e:
        print(f"Warning: Could not load meta.pkl for calibration. Calibration will use random data. {e}")
        return None, None


# ==============================================================================
# OMEGAQUANT ALGORITHMS
# ==============================================================================

def sensitivity_quantize(w, n_bits, h_diag, gs=128):
    """
    TMG-Q Phase 3: Asymmetric + Outlier Clipped + Hessian-guided Quantization
    """
    q_levels = (2**n_bits) - 1
    rows, cols = w.shape
    w_q = torch.zeros_like(w)
    
    for cs in range(0, cols, gs):
        ce = min(cs+gs, cols)
        block = w[:, cs:ce].clone()
        
        # 1. Dynamic Fast Outlier Shielding (Clipping at 3.5 Sigma)
        b_mean = block.mean(dim=1, keepdim=True)
        b_std = block.std(dim=1, keepdim=True).clamp(min=1e-8)
        lower_bound = b_mean - (3.5 * b_std)
        upper_bound = b_mean + (3.5 * b_std)
        block = torch.where(block > upper_bound, upper_bound, block)
        block = torch.where(block < lower_bound, lower_bound, block)
        
        # 2. Tanh-based Soft-Smoothing for Non-Linear Distribution
        block = torch.tanh(block) * (b_std * 3.5) if False else block
        
        # 3. Asymmetric Zero-Point Scaling
        b_min = block.min(dim=1, keepdim=True).values
        b_max = block.max(dim=1, keepdim=True).values
        scale = (b_max - b_min).clamp(min=1e-8) / q_levels
        zero_point = torch.round(-b_min / scale)
        
        ws = (block / scale) + zero_point
        
        # 4. Hessian-Guided Sensitivity Check
        if h_diag is not None and ce <= h_diag.shape[0]:
            hb = h_diag[cs:ce].unsqueeze(0).clamp(min=1e-10)
            wf = torch.floor(ws)
            wc = torch.ceil(ws)
            wr = torch.round(ws)
            
            ef = ((ws - wf)**2) * hb
            er = ((ws - wr)**2) * hb
            ec = ((ws - wc)**2) * hb
            
            opts = torch.stack([wf, wr, wc], dim=0)
            errs = torch.stack([ef, er, ec], dim=0)
            best = errs.argmin(dim=0)
            wr = torch.gather(opts, 0, best.unsqueeze(0)).squeeze(0)
        else:
            wr = torch.round(ws)
            
        wr = torch.clamp(wr, 0, q_levels)
        w_q[:, cs:ce] = (wr - zero_point) * scale

    return w_q

def error_diffusion(w_orig, w_q, h_diag, n_waves=3, gs=128):
    wc = w_q.clone()
    for wave in range(n_waves):
        res = w_orig - wc
        if h_diag is not None:
            imp = h_diag.sqrt().clamp(min=1e-8)
            imp = imp / imp.max()
            if imp.shape[0] == res.shape[1]:
                res = res * imp.unsqueeze(0)
        rb = max(2, 3-wave); rqmax = 2**(rb-1)-1; rqmin = -rqmax-1
        for cs in range(0, res.shape[1], gs):
            ce = min(cs+gs, res.shape[1])
            blk = res[:, cs:ce]
            sc = blk.abs().max(1, keepdim=True).values.clamp(min=1e-8)/rqmax
            res[:, cs:ce] = torch.clamp(torch.round(blk/sc), rqmin, rqmax)*sc
        if h_diag is not None and imp.shape[0] == res.shape[1]:
            res = res / imp.unsqueeze(0)
        wc = wc + res
    return wc

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    import functools
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def apply_omegaquant(model, bits, h_diag_all, device):
    """Apply TMG-Q Ultra inplace and pack into physical INT32 limits."""
    total_layers = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    current_layer = 1
    
    linear_names = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    
    for n in linear_names:
        m = rgetattr(model, n)
        w = m.weight.data
        print(f"  [{current_layer}/{total_layers}] Packing {n} ({w.shape[0]}x{w.shape[1]})", end='\r')
        orig = w.clone()
        hd = h_diag_all.get(n)
        
        wq = sensitivity_quantize(w, bits, hd)
        wq = error_diffusion(orig, wq, hd, n_waves=3)
        
        # TMG-Q HARDWARE PACKING: Destroy FP16, inject INT32 QuantizedLinear
        qlayer = QuantizedLinear(m.in_features, m.out_features, bias=m.bias is not None, gs=128)
        qlayer.pack_from_float(wq, m.bias.data if m.bias is not None else None, n_bits=bits)
        
        pre, _, post = n.rpartition('.')
        parent = rgetattr(model, pre) if pre else model
        setattr(parent, post, qlayer)
        
        current_layer += 1
            
    print(f"\n  Done Quantizing & Packing {total_layers} layers!")
    return model


def calibrate_model(model, stoi, device, n_tasks=50):
    """Extract Hessian weights using calibration dataset."""
    if stoi is None:
        # Fallback to random tokens if curriculum not available
        h_diag = {}
        id2name = {id(m): n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
        def hook(mod, inp, out):
            nm = id2name.get(id(mod))
            if nm is None: return
            x = inp[0].detach().float()
            if x.dim() == 3: x = x.reshape(-1, x.size(-1))
            hd = (x**2).mean(0)
            if nm in h_diag: h_diag[nm] += hd
            else: h_diag[nm] = hd.clone()
            
        handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
        model.eval()
        with torch.no_grad():
            for _ in range(n_tasks):
                x = torch.randint(0, 65, (1, 64), device=device)
                model(x)
        for h in handles: h.remove()
        for nm in h_diag: h_diag[nm] /= n_tasks
        return h_diag

    # Proper calibration using the NanoGPT Custom Curriculum
    examples = build_examples(count=n_tasks, seed=42, difficulty=2)
    h_diag = {}
    id2name = {id(m): n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    
    def hook(mod, inp, out):
        nm = id2name.get(id(mod))
        if nm is None: return
        x = inp[0].detach().float()
        if x.dim() == 3: x = x.reshape(-1, x.size(-1))
        hd = (x**2).mean(0)
        if nm in h_diag: h_diag[nm] += hd
        else: h_diag[nm] = hd.clone()
        
    handles = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Linear)]
    model.eval()
    
    with torch.no_grad():
        for ex in examples:
            prompt_text = build_prompt(ex)
            ids = [stoi[ch] for ch in prompt_text]
            x = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
            model(x)
            
    for h in handles: h.remove()
    for nm in h_diag: h_diag[nm] /= n_tasks
    return h_diag


def num_params_mb(model, bits_per_weight=16):
    total = sum(p.numel() for p in model.parameters())
    lp = sum(m.weight.numel() for m in model.modules() if isinstance(m, nn.Linear))
    op = total - lp
    return (lp * bits_per_weight / 8 + op * 2) / 1e6


def main():
    parser = argparse.ArgumentParser(description="OmegaQuant CLI - Local LLM Compression")
    parser.add_argument("--in-ckpt", type=str, required=True, help="Path to input PyTorch checkpoint (ckpt.pt)")
    parser.add_argument("--out-ckpt", type=str, required=True, help="Path to save the quantized checkpoint")
    parser.add_argument("--bits", type=int, choices=[2, 3, 4, 8], default=3, help="Quantization bits (e.g., 3)")
    parser.add_argument("--calib-tasks", type=int, default=50, help="Number of calibration tasks for Hessian analysis")
    args = parser.parse_args()

    in_path = Path(args.in_ckpt)
    out_path = Path(args.out_ckpt)

    if not in_path.exists():
        print(f"Error: Input checkpoint missing at {in_path}")
        sys.exit(1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"============================================================")
    print(f" T M G - Q   U L T R A    [Device: {device.upper()}]")
    print(f" Phase 3: Strict INT Asymmetric Quantization")
    print(f"============================================================")
    print(f"Target : {args.bits}-bit quantization")
    print(f"Input  : {in_path.name}")
    print(f"Output : {out_path.name}")
    print(f"============================================================\n")

    print("[1/4] Loading model and metadata...")
    model, checkpoint = load_model_from_checkpoint(in_path, device)
    stoi, _ = load_meta()
    
    orig_mb = num_params_mb(model, 16)
    quant_mb = num_params_mb(model, args.bits)
    print(f"   -> Original logical size: ~{orig_mb:.1f} MB (FP16)")
    print(f"   -> Target logical size: ~{quant_mb:.1f} MB ({args.bits}-bit)")

    print(f"\n[2/4] Calibrating Hessian Sensitivity ({args.calib_tasks} tasks)...")
    h_diag = calibrate_model(model, stoi, device, n_tasks=args.calib_tasks)
    print(f"   -> Calibrated {len(h_diag)} matrices.")

    print(f"\n[3/4] Applying SVD + Hessian Error Diffusion...")
    model = apply_omegaquant(model, args.bits, h_diag, device)

    print(f"\n[4/4] Saving compressed model checkpoint...")
    # Update the checkpoint dictionary with quantized float16 weights
    checkpoint["model"] = model.state_dict()
    # Add metadata about quantization
    checkpoint["omegaquant"] = {"bits": args.bits, "technique": "svd_hessian_v2"}
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_path)
    
    print(f"   -> Success! Saved to {out_path}")
    print(f"============================================================")


if __name__ == "__main__":
    main()
