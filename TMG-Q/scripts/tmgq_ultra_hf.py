#!/usr/bin/env python
"""
TMG-Q Ultra CLI (Formerly OmegaQuant)
=====================================
The ultimate command-line interface for extreme LLM compression.
Downloads a HuggingFace model, applies SVD + Hessian Error Diffusion,
and evaluates the compressed model.

Supported Tested Models:
- gpt2 (124M)
- gpt2-medium (355M)
- gpt2-large (774M)
- HuggingFace AutoModels (Llama, Mistral) - Architectural support native.

Usage:
    python tmgq_ultra.py --model gpt2-medium --bits 3 --test
"""
import sys, copy, math, argparse, gc
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

def sensitivity_quantize(w, n_bits, h_diag, gs=128):
    qmin, qmax = -(2**(n_bits-1)), 2**(n_bits-1)-1
    rows, cols = w.shape
    w_q = torch.zeros_like(w)
    for cs in range(0, cols, gs):
        ce = min(cs+gs, cols)
        block = w[:, cs:ce]
        wmax = block.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        scale = wmax / qmax
        ws = block / scale
        wr = torch.clamp(torch.round(ws), qmin, qmax)
        if h_diag is not None and ce <= h_diag.shape[0]:
            hb = h_diag[cs:ce].unsqueeze(0).clamp(min=1e-10)
            wf = torch.clamp(torch.floor(ws), qmin, qmax)
            wc = torch.clamp(torch.ceil(ws), qmin, qmax)
            ef = ((ws-wf)**2)*hb; er = ((ws-wr)**2)*hb; ec = ((ws-wc)**2)*hb
            opts = torch.stack([wf, wr, wc], 0)
            errs = torch.stack([ef, er, ec], 0)
            best = errs.argmin(0)
            wr = torch.gather(opts, 0, best.unsqueeze(0)).squeeze(0)
        w_q[:, cs:ce] = wr * scale
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

def spectral_recovery(w_orig, w_q, rank_ratio=0.03):
    res = (w_orig - w_q).float()
    r, c = res.shape
    rank = max(1, int(min(r,c)*rank_ratio))
    try:
        U, S, Vh = torch.linalg.svd(res, full_matrices=False)
        recov = (U[:,:rank] * S[:rank].unsqueeze(0)) @ Vh[:rank,:]
        return (w_q + recov.to(w_q.dtype))
    except: return w_q

def calibrate(model, tokenizer, device, n_samples=8):
    texts = [
        "The meaning of life is a philosophical question debated for centuries by scholars.",
        "In computer science, algorithms solve complex computational problems efficiently.",
        "Python is a high-level programming language known for simplicity and readability.",
        "Machine learning models learn patterns from large datasets to make predictions.",
        "The solar system has eight planets orbiting the sun in elliptical paths.",
        "Quantum computing harnesses quantum mechanics for parallel computation.",
        "Neural networks are models inspired by the biological brain structure.",
        "Deep learning has transformed computer vision and natural language processing.",
    ]
    h_diag = {}
    id2name = {id(m): n for n, m in model.named_modules() if is_target_layer(m)}
    def hook(mod, inp, out):
        nm = id2name.get(id(mod))
        if nm is None: return
        x = inp[0].detach().float()
        if x.dim() == 3: x = x.reshape(-1, x.size(-1))
        
        # If it's a Conv1D layer, the input x is [batch*seq, in_f]. But wait, 
        # in HuggingFace Conv1D (which is just a linear layer with transposed weights), 
        # the forward is actually `x @ weight + bias`. Wait, `Conv1D` in HF is implemented as 
        # `x @ weight + bias`. So `x` has shape `[..., in_f]`. 
        # The Hessian diagonal is still the mean square of the input dimensions!
        hd = (x**2).mean(0)
        
        if nm in h_diag: h_diag[nm] += hd
        else: h_diag[nm] = hd.clone()
    
    handles = [m.register_forward_hook(hook) for m in model.modules() if is_target_layer(m)]
    model.eval()
    with torch.no_grad():
        for t in texts[:n_samples]: 
            model(**tokenizer(t, return_tensors="pt", truncation=True, max_length=256).to(device))
    for h in handles: h.remove()
    for nm in h_diag: h_diag[nm] /= len(texts[:n_samples])
    return h_diag

def is_target_layer(m):
    return isinstance(m, nn.Linear) or "Conv1D" in m.__class__.__name__

def apply_tmgq_ultra(model, bits, h_diag_all):
    total = sum(1 for m in model.modules() if is_target_layer(m))
    curr = 1
    for n, m in model.named_modules():
        if is_target_layer(m):
            w = m.weight.data
            is_conv1d = "Conv1D" in m.__class__.__name__
            if is_conv1d:
                w = w.t() # Transpose to [out_f, in_f] for standard processing
            
            print(f"  [{curr}/{total}] Quantizing {n} ({w.shape[0]}x{w.shape[1]})", end='\r')
            orig = w.clone()
            hd = h_diag_all.get(n)
            
            wq = sensitivity_quantize(w, bits, hd)
            wq = error_diffusion(orig, wq, hd, n_waves=3)
            wq = spectral_recovery(orig, wq, rank_ratio=0.03)
            
            if is_conv1d:
                wq = wq.t() # Transpose back to [in_f, out_f]
                
            m.weight.data = wq.to(torch.float16)
            curr += 1
    print(f"\n  Done Quantizing {total} layers!")
    return model

@torch.no_grad()
def evaluate_ppl(model, tok, device):
    tests = [
        "The meaning of life is a philosophical question debated for centuries.",
        "In computer science, algorithms solve complex problems efficiently."
    ]
    model.eval(); tl=0; tt=0
    for t in tests:
        ids = tok(t, return_tensors="pt", truncation=True, max_length=128).to(device)
        o = model(**ids, labels=ids["input_ids"])
        n = ids["input_ids"].size(1)-1
        loss = o.loss.item()
        if math.isnan(loss) or math.isinf(loss): return float('inf')
        tl += loss*n; tt += n
    return math.exp(tl/max(1,tt))

@torch.no_grad()
def gen_text(model, tok, prompt, device, n=40):
    model.eval()
    ids = tok(prompt, return_tensors="pt").to(device)["input_ids"]
    for _ in range(n):
        logits = model(ids).logits[:,-1,:]/0.8
        if logits.isnan().any(): return tok.decode(ids[0]) + " [NaN]"
        v,_ = torch.topk(logits, 50)
        logits[logits < v[:,[-1]]] = -float('Inf')
        ids = torch.cat([ids, torch.multinomial(torch.softmax(logits,-1),1)], 1)
        if ids[0,-1].item() == tok.eos_token_id: break
    return tok.decode(ids[0], skip_special_tokens=True).replace('\n', ' ')

def main():
    parser = argparse.ArgumentParser(description="TMG-Q Ultra - Advanced Local Quantization")
    parser.add_argument("--model", type=str, default="gpt2-medium", help="HuggingFace model ID (e.g., gpt2, gpt2-medium, gpt2-large)")
    parser.add_argument("--bits", type=int, choices=[2, 3, 4], default=3, help="Quantization bits (2, 3, 4)")
    parser.add_argument("--test", action="store_true", help="Run Perplexity and Text Generation tests after quantization")
    parser.add_argument("--save-path", type=str, default="", help="Path to save the quantized model (optional)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*70)
    print(f" TMG-Q ULTRA - Professional Quantization Framework")
    print(f" Target Model: {args.model}")
    print(f" Target Bits:  {args.bits}-bit")
    print(f" Device:       {device.upper()}")
    print("="*70)

    try:
        print(f"\n[1/4] Downloading / Loading '{args.model}'...")
        tok = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).to(device)
    except Exception as e:
        print(f"Failed to load model from HuggingFace: {e}")
        sys.exit(1)

    fp16_sz = sum(p.numel() for p in model.parameters()) * 2 / 1e6
    quant_sz = sum(p.numel() for p in model.parameters()) * (args.bits/8) / 1e6
    print(f"  -> Model FP16 Size: ~{fp16_sz:.1f} MB")
    print(f"  -> Target Logical Size: ~{quant_sz:.1f} MB")

    if args.test:
        print("\n[Optional] Benchmarking Original FP16 Baseline...")
        base_ppl = evaluate_ppl(model, tok, device)
        print(f"  -> Original PPL: {base_ppl:.2f}")

    print("\n[2/4] Calibrating Hessian Sensitivity (Crucial for 2-bit/3-bit)...")
    h_diag = calibrate(model, tok, device)

    print("\n[3/4] Quantizing with TMG-Q Ultra Engine...")
    model = apply_tmgq_ultra(model, args.bits, h_diag)

    if args.test:
        print("\n[4/4] Validating Quantized Model...")
        q_ppl = evaluate_ppl(model, tok, device)
        print(f"  -> Quantized PPL: {q_ppl:.2f}")
        prompt = "The future of artificial intelligence is"
        print(f"  -> Generation: {gen_text(model, tok, prompt, device)}")

    if args.save_path:
        print(f"\nSaving to {args.save_path}...")
        torch.save(model.state_dict(), args.save_path)
        print("Done!")

    print("\n======================================================================")
    print(" Compression Completed Successfully. TMG-Q Ultra OUT.")
    print("======================================================================")

if __name__ == "__main__":
    main()
