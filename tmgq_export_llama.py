import torch
import torch.nn as nn
from tmgq_transformers_compat import disable_broken_torchvision

disable_broken_torchvision()
from transformers import AutoModelForCausalLM, AutoTokenizer
from tmgq_packer import QuantizedEmbedding, QuantizedLinear, QuantizedTiedLMHead, pack_nbit
from tmgq_budget_optimizer import QuantizationCandidate, estimate_payload_bytes, optimize_layer_budget
import argparse
import gc
import math

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    import functools
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def is_projection_module(module):
    return isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D"

def projection_weight(module):
    weight = module.weight.data
    return weight.t() if module.__class__.__name__ == "Conv1D" else weight

def transformer_layer_index(name):
    parts = name.split(".")
    for marker in ("layers", "h", "blocks"):
        if marker in parts:
            pos = parts.index(marker)
            if pos + 1 < len(parts) and parts[pos + 1].isdigit():
                return int(parts[pos + 1])
    return None

def resolve_layer_quantization(name, default_bits, default_quantizer, mixed_policy, num_hidden_layers):
    if mixed_policy == "none":
        return default_bits, default_quantizer
    if mixed_policy in ("adaptive", "budget"):
        return default_bits, default_quantizer

    layer_idx = transformer_layer_index(name)
    if layer_idx is None or num_hidden_layers is None:
        return default_bits, default_quantizer

    boundary = layer_idx < 2 or layer_idx >= max(0, num_hidden_layers - 2)
    if mixed_policy == "balanced":
        if boundary:
            return 4, "linear"
        return 3, "codebook"

    if mixed_policy == "aggressive":
        if boundary:
            return 3, "codebook"
        if ".mlp." in name or "feed_forward" in name:
            return 2, "codebook"
        return 3, "codebook"

    raise ValueError(f"Unknown mixed policy: {mixed_policy}")

def quantization_nmse(w, dequantized, h_diag=None):
    error = (w.float() - dequantized.float()) ** 2
    energy = w.float() ** 2
    if h_diag is not None and h_diag.numel() >= w.shape[1]:
        weights = h_diag[:w.shape[1]].float().clamp(min=1e-10).view(1, -1).to(w.device)
        error = error * weights
        energy = energy * weights
    return (error.sum() / energy.sum().clamp(min=1e-12)).item()

def quantize_schema(w, bits, quantizer, group_size, h_diag, clip_search, codebook_iters):
    if quantizer == "codebook":
        limits, codebooks = codebook_quantize_packable(
            w,
            bits,
            gs=group_size,
            h_diag=h_diag,
            codebook_iters=codebook_iters,
        )
        scales = torch.empty(0, dtype=torch.float16, device=w.device)
        zeros = torch.empty(0, dtype=torch.float16, device=w.device)
    elif quantizer == "linear":
        limits, scales, zeros = sensitivity_quantize_packable(
            w,
            bits,
            gs=group_size,
            clip_search=clip_search,
            h_diag=h_diag,
        )
        codebooks = torch.empty(0, dtype=torch.float16, device=w.device)
    else:
        raise ValueError("quantizer must be 'linear' or 'codebook'")
    return limits, scales, zeros, codebooks

def adaptive_quantize_schema(
    w,
    group_size,
    h_diag,
    clip_search,
    codebook_iters,
    threshold_2bit,
    threshold_3bit,
    minimum_bits=2,
):
    candidates = tuple(
        candidate
        for candidate in (
            (2, "codebook", threshold_2bit),
            (3, "codebook", threshold_3bit),
            (4, "linear", None),
        )
        if candidate[0] >= minimum_bits
    )
    metrics = {}
    selected = None
    with torch.no_grad():
        for bits, quantizer, threshold in candidates:
            limits, scales, zeros, codebooks = quantize_schema(
                w,
                bits,
                quantizer,
                group_size,
                h_diag,
                clip_search,
                codebook_iters,
            )
            deq = dequantize_schema(limits, scales, zeros, group_size, codebooks=codebooks)
            nmse = quantization_nmse(w, deq, h_diag=h_diag)
            metrics[f"{bits}bit_{quantizer}"] = nmse
            del deq
            if selected is None or nmse < selected[0]:
                selected = (nmse, bits, quantizer)
            if threshold is None or nmse <= threshold:
                if threshold is not None:
                    return bits, quantizer, limits, scales, zeros, codebooks, metrics
            del limits, scales, zeros, codebooks
            gc.collect()

    if selected is None:
        raise RuntimeError("Adaptive quantization did not produce a candidate")
    _, bits, quantizer = selected
    limits, scales, zeros, codebooks = quantize_schema(
        w,
        bits,
        quantizer,
        group_size,
        h_diag,
        clip_search,
        codebook_iters,
    )
    return bits, quantizer, limits, scales, zeros, codebooks, metrics

def profile_quantization_candidates(w, group_size, h_diag, clip_search, codebook_iters):
    candidates = ((2, "codebook"), (3, "codebook"), (4, "linear"))
    metrics = {}
    with torch.no_grad():
        for bits, quantizer in candidates:
            limits, scales, zeros, codebooks = quantize_schema(
                w,
                bits,
                quantizer,
                group_size,
                h_diag,
                clip_search,
                codebook_iters,
            )
            deq = dequantize_schema(limits, scales, zeros, group_size, codebooks=codebooks)
            metrics[(bits, quantizer)] = quantization_nmse(w, deq, h_diag=h_diag)
            del limits, scales, zeros, codebooks, deq
            gc.collect()
    return metrics

def select_probe_rows(weight, rows):
    count = min(weight.shape[0], max(1, rows))
    if count == weight.shape[0]:
        return weight
    indices = torch.linspace(0, weight.shape[0] - 1, count, device=weight.device).round().long()
    return weight.index_select(0, indices)

def adaptive_quantize_with_probe(
    w,
    group_size,
    h_diag,
    clip_search,
    codebook_iters,
    threshold_2bit,
    threshold_3bit,
    probe_rows=256,
    minimum_bits=2,
):
    probe = select_probe_rows(w, probe_rows)

    (
        bits,
        quantizer,
        probe_limits,
        probe_scales,
        probe_zeros,
        probe_codebooks,
        metrics,
    ) = adaptive_quantize_schema(
        probe,
        group_size,
        h_diag,
        clip_search,
        codebook_iters,
        threshold_2bit,
        threshold_3bit,
        minimum_bits=minimum_bits,
    )
    del probe_limits, probe_scales, probe_zeros, probe_codebooks
    if probe is not w:
        del probe
    gc.collect()

    limits, scales, zeros, codebooks = quantize_schema(
        w,
        bits,
        quantizer,
        group_size,
        h_diag,
        clip_search,
        codebook_iters,
    )
    return bits, quantizer, limits, scales, zeros, codebooks, metrics

def _quantize_block(block, n_bits, sigma_clip=None, h_block=None):
    q_levels = (2**n_bits) - 1
    original = block.float()
    work = block.float()
    if sigma_clip is not None:
        mean = work.mean(dim=1, keepdim=True)
        std = work.std(dim=1, keepdim=True, unbiased=False).clamp(min=1e-8)
        work = work.clamp(mean - sigma_clip * std, mean + sigma_clip * std)

    b_min = work.min(dim=1, keepdim=True).values
    b_max = work.max(dim=1, keepdim=True).values
    scale = (b_max - b_min).clamp(min=1e-8) / q_levels
    scaled = (work - b_min) / scale

    if h_block is None:
        limits = torch.clamp(torch.round(scaled), 0, q_levels).to(torch.int32)
    else:
        h = h_block.float().view(1, -1).clamp(min=1e-10)
        floor = torch.floor(scaled)
        round_ = torch.round(scaled)
        ceil = torch.ceil(scaled)
        opts = torch.stack([floor, round_, ceil], dim=0).clamp(0, q_levels)
        deq_opts = opts * scale.unsqueeze(0) + b_min.unsqueeze(0)
        errs = ((original.unsqueeze(0) - deq_opts) ** 2) * h.unsqueeze(0)
        best = errs.argmin(dim=0).unsqueeze(0)
        limits = torch.gather(opts, 0, best).squeeze(0).to(torch.int32)

    deq = limits.float() * scale + b_min
    return limits, scale, b_min, deq

def sensitivity_quantize_packable(w, n_bits, gs=128, clip_search=True, h_diag=None):
    """Extract INT limits and FP16 scales with row-wise outlier-aware clipping search."""
    q_levels = (2**n_bits) - 1
    rows, cols = w.shape
    num_blocks = math.ceil(cols / gs)
    
    limits_int = torch.zeros_like(w, dtype=torch.int32)
    scales = torch.zeros((rows, num_blocks), dtype=torch.float16, device=w.device)
    zeros = torch.zeros((rows, num_blocks), dtype=torch.float16, device=w.device)
    
    for i, cs in enumerate(range(0, cols, gs)):
        ce = min(cs+gs, cols)
        block = w[:, cs:ce].clone()
        h_block = h_diag[cs:ce] if h_diag is not None and h_diag.numel() >= ce else None

        candidates = (None, 4.0, 3.5, 3.0, 2.5) if clip_search else (None,)
        best_limits = None
        best_scale = None
        best_zero = None
        best_err = None

        for sigma in candidates:
            cand_limits, cand_scale, cand_zero, deq = _quantize_block(
                block,
                n_bits,
                sigma_clip=sigma,
                h_block=h_block,
            )
            if h_block is None:
                err = ((block.float() - deq) ** 2).mean(dim=1, keepdim=True)
            else:
                h = h_block.float().view(1, -1).clamp(min=1e-10)
                err = (((block.float() - deq) ** 2) * h).mean(dim=1, keepdim=True)
            if best_err is None:
                best_limits, best_scale, best_zero, best_err = cand_limits, cand_scale, cand_zero, err
                continue

            take = err < best_err
            best_limits = torch.where(take, cand_limits, best_limits)
            best_scale = torch.where(take, cand_scale, best_scale)
            best_zero = torch.where(take, cand_zero, best_zero)
            best_err = torch.where(take, err, best_err)

        limits_int[:, cs:ce] = best_limits
        scales[:, i:i+1] = best_scale.to(torch.float16)
        zeros[:, i:i+1] = best_zero.to(torch.float16)
        
    return limits_int, scales, zeros

def _codebook_quantize_block(block, n_bits, h_block=None, iters=6):
    levels = 2 ** n_bits
    work = block.float()
    rows, cols = work.shape

    b_min = work.min(dim=1, keepdim=True).values
    b_max = work.max(dim=1, keepdim=True).values
    fractions = torch.linspace(0.0, 1.0, levels, device=work.device, dtype=work.dtype).view(1, levels)
    codebook = b_min + (b_max - b_min).clamp(min=1e-8) * fractions
    weights = h_block.float().view(1, cols) if h_block is not None else None

    assignments = torch.zeros((rows, cols), dtype=torch.long, device=work.device)
    for _ in range(max(1, iters)):
        distances = (work.unsqueeze(-1) - codebook.unsqueeze(1)) ** 2
        assignments = distances.argmin(dim=-1)
        for level in range(levels):
            mask = (assignments == level).float()
            if weights is not None:
                weighted = mask * weights
                denom = weighted.sum(dim=1).clamp(min=1e-8)
                updated = (work * weighted).sum(dim=1) / denom
                has_values = weighted.sum(dim=1) > 0
            else:
                denom = mask.sum(dim=1).clamp(min=1e-8)
                updated = (work * mask).sum(dim=1) / denom
                has_values = mask.sum(dim=1) > 0
            codebook[:, level] = torch.where(has_values, updated, codebook[:, level])

    distances = (work.unsqueeze(-1) - codebook.unsqueeze(1)) ** 2
    assignments = distances.argmin(dim=-1).to(torch.int32)
    return assignments, codebook.to(torch.float16)

def codebook_quantize_packable(w, n_bits, gs=128, h_diag=None, codebook_iters=6, row_chunk_size=512):
    rows, cols = w.shape
    levels = 2 ** n_bits
    num_blocks = math.ceil(cols / gs)
    limits_int = torch.zeros_like(w, dtype=torch.int32)
    codebooks = torch.zeros((rows, num_blocks, levels), dtype=torch.float16, device=w.device)

    for i, cs in enumerate(range(0, cols, gs)):
        ce = min(cs + gs, cols)
        h_block = h_diag[cs:ce] if h_diag is not None and h_diag.numel() >= ce else None
        for rs in range(0, rows, row_chunk_size):
            re = min(rs + row_chunk_size, rows)
            block = w[rs:re, cs:ce]
            assignments, block_codebooks = _codebook_quantize_block(
                block,
                n_bits,
                h_block=h_block,
                iters=codebook_iters,
            )
            limits_int[rs:re, cs:ce] = assignments
            codebooks[rs:re, i, :] = block_codebooks
            del assignments, block_codebooks

    return limits_int, codebooks

def dequantize_schema(limits_int, scales, zeros, gs, codebooks=None):
    cols = limits_int.shape[1]
    group_ids = torch.arange(cols, device=limits_int.device) // gs
    if codebooks is not None and codebooks.numel() > 0:
        group_codebooks = codebooks[:, group_ids, :].float()
        return torch.gather(group_codebooks, 2, limits_int.to(torch.long).unsqueeze(-1)).squeeze(-1)
    return limits_int.float() * scales[:, group_ids].float() + zeros[:, group_ids].float()

def build_sparse_residual(w, limits_int, scales, zeros, gs, outlier_fraction=0.0, h_diag=None, codebooks=None):
    if outlier_fraction <= 0:
        device = w.device
        return (
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.int32, device=device),
            torch.empty(0, dtype=torch.float16, device=device),
        )

    rows, cols = w.shape
    deq = dequantize_schema(limits_int, scales, zeros, gs, codebooks=codebooks)
    residual = w.float() - deq
    score = residual.abs()
    if h_diag is not None and h_diag.numel() >= cols:
        score = score * h_diag[:cols].float().sqrt().view(1, -1).to(score.device)

    k = min(score.numel(), max(1, int(score.numel() * outlier_fraction)))
    flat_idx = torch.topk(score.flatten(), k=k, largest=True).indices
    outlier_rows = (flat_idx // cols).to(torch.int32)
    outlier_cols = (flat_idx % cols).to(torch.int32)
    outlier_values = residual.flatten()[flat_idx].to(torch.float16)
    return outlier_rows, outlier_cols, outlier_values

def build_low_rank_residual(w, limits_int, scales, zeros, gs, outlier_rows=None, outlier_cols=None, svd_rank=0, codebooks=None):
    if svd_rank <= 0:
        device = w.device
        return (
            torch.empty(0, dtype=torch.float16, device=device),
            torch.empty(0, dtype=torch.float16, device=device),
            torch.empty(0, dtype=torch.float16, device=device),
        )

    rows, cols = w.shape
    rank = min(svd_rank, min(rows, cols))
    deq = dequantize_schema(limits_int, scales, zeros, gs, codebooks=codebooks)
    residual = w.float() - deq
    if outlier_rows is not None and outlier_cols is not None and outlier_rows.numel() > 0:
        residual[outlier_rows.to(torch.long), outlier_cols.to(torch.long)] = 0.0

    u, s, vh = torch.linalg.svd(residual.cpu(), full_matrices=False)
    return (
        u[:, :rank].to(torch.float16),
        s[:rank].to(torch.float16),
        vh[:rank, :].to(torch.float16),
    )

def get_calibration_chunks(tokenizer, max_length=256, max_samples=64):
    if load_dataset is None:
        raise RuntimeError("Calibration requires the 'datasets' package. Install it with: pip install datasets")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in dataset["text"] if t.strip())
    tokenized = tokenizer(text, return_tensors="pt")["input_ids"][0]

    chunks = []
    for start in range(0, len(tokenized) - max_length, max_length):
        if len(chunks) >= max_samples:
            break
        chunks.append(tokenized[start:start + max_length].unsqueeze(0))
    return chunks

def collect_hessian_diagonal(model, tokenizer, device, max_length=256, max_samples=64):
    """Approximate the Hessian diagonal per Linear input channel using E[x^2]."""
    chunks = get_calibration_chunks(tokenizer, max_length=max_length, max_samples=max_samples)
    h_diag = {}
    module_names = {id(m): n for n, m in model.named_modules() if is_projection_module(m)}

    def hook(module, inputs, output):
        name = module_names.get(id(module))
        if name is None or not inputs:
            return
        x = inputs[0].detach().float()
        if x.dim() == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.dim() > 3:
            x = x.reshape(-1, x.shape[-1])
        current = (x ** 2).mean(dim=0).cpu()
        h_diag[name] = h_diag.get(name, torch.zeros_like(current)) + current

    handles = [m.register_forward_hook(hook) for m in model.modules() if is_projection_module(m)]
    model.eval()
    with torch.no_grad():
        for idx, chunk in enumerate(chunks, 1):
            print(f"  Calibration block {idx}/{len(chunks)}", end="\r")
            model(chunk.to(device))

    for handle in handles:
        handle.remove()
    for name in h_diag:
        h_diag[name] /= max(1, len(chunks))
    print(f"\nCollected Hessian proxies for {len(h_diag)} linear layers.")
    return h_diag

def export_huggingface_model(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    bits=4,
    export_path="TinyLlama_4bit_TMGQ.pt",
    group_size=128,
    clip_search=True,
    calibrate=False,
    calib_samples=64,
    calib_length=256,
    outlier_fraction=0.0,
    svd_rank=0,
    quantizer="linear",
    codebook_iters=6,
    mixed_policy="none",
    adaptive_2bit_nmse=0.06,
    adaptive_3bit_nmse=0.02,
    adaptive_probe_rows=256,
    quant_device="auto",
    target_ratio=3.5,
    quantize_embeddings=False,
    embedding_bits=4,
    quantize_lm_head=False,
    embedding_svd_rank=0,
    adaptive_min_bits=2,
):
    if bits not in (2, 3, 4):
        raise ValueError("bits must be 2, 3, or 4")

    print(f"Loading base bfloat16 model: {model_name} from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    if quant_device == "auto":
        quant_device = "cuda" if torch.cuda.is_available() else "cpu"
    if quant_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--quant-device cuda was requested, but CUDA is unavailable")
    device = quant_device
    print(f"Quantization device: {device}")
    h_diag_all = {}

    if calibrate:
        print("\nRunning calibration pass for Hessian-guided rounding...")
        model = model.to(device)
        h_diag_all = collect_hessian_diagonal(
            model,
            tokenizer,
            device,
            max_length=calib_length,
            max_samples=calib_samples,
        )
        model = model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    total_layers = sum(1 for m in model.modules() if is_projection_module(m))
    current_layer = 1
    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    
    print("\nStarting TMG-Q Ultra Advanced Packing Sequence...")
    linear_names = [n for n, m in model.named_modules() if is_projection_module(m)]
    budget_plan = None
    if mixed_policy == "budget":
        print(f"Profiling layer candidates for a {target_ratio:.2f}x payload target...")
        candidates = []
        fp16_bytes = 0
        for profile_index, n in enumerate(linear_names, 1):
            if "lm_head" in n:
                continue
            m = rgetattr(model, n)
            w = projection_weight(m).to(device, non_blocking=True)
            h_diag = h_diag_all.get(n)
            if h_diag is not None:
                h_diag = h_diag.to(device, non_blocking=True)
            sample = select_probe_rows(w, adaptive_probe_rows)
            metrics = profile_quantization_candidates(
                sample,
                group_size,
                h_diag,
                clip_search,
                codebook_iters,
            )
            fp16_bytes += w.numel() * 2
            for (candidate_bits, candidate_quantizer), nmse in metrics.items():
                candidates.append(
                    QuantizationCandidate(
                        layer=n,
                        bits=candidate_bits,
                        quantizer=candidate_quantizer,
                        loss=nmse * w.numel(),
                        payload_bytes=estimate_payload_bytes(
                            w.shape[0],
                            w.shape[1],
                            candidate_bits,
                            candidate_quantizer,
                            group_size,
                            outlier_fraction,
                        ),
                    )
                )
            print(f"  Profiled {profile_index}/{len(linear_names)}: {n}", end="\r")
            del m, w, h_diag, sample
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        budget_plan = optimize_layer_budget(candidates, fp16_bytes, target_ratio)
        bit_counts = {}
        for choice in budget_plan.choices.values():
            bit_counts[choice.bits] = bit_counts.get(choice.bits, 0) + 1
        print(
            f"\nBudget plan: {budget_plan.ratio:.2f}x, "
            f"{budget_plan.payload_bytes / 1024**2:.1f} MiB, layers {bit_counts}"
        )
    
    for n in linear_names:
        m = rgetattr(model, n)
        w = projection_weight(m).to(device, non_blocking=True)
        h_diag = h_diag_all.get(n)
        if h_diag is not None:
            h_diag = h_diag.to(device, non_blocking=True)
        if "lm_head" in n:
             # Skip output embeddings to preserve reasoning accuracy
             del w, h_diag
             current_layer += 1
             continue
             
        if mixed_policy == "adaptive":
            (
                layer_bits,
                layer_quantizer,
                limits_int,
                scales,
                zeros,
                codebooks,
                adaptive_metrics,
            ) = adaptive_quantize_with_probe(
                w,
                group_size,
                h_diag,
                clip_search,
                codebook_iters,
                adaptive_2bit_nmse,
                adaptive_3bit_nmse,
                probe_rows=adaptive_probe_rows,
                minimum_bits=adaptive_min_bits,
            )
            metric_text = ", ".join(f"{key}={value:.4g}" for key, value in adaptive_metrics.items())
        elif mixed_policy == "budget":
            choice = budget_plan.choices[n]
            layer_bits, layer_quantizer = choice.bits, choice.quantizer
            adaptive_metrics = None
            metric_text = f"budget loss={choice.loss:.4g}"
        else:
            layer_bits, layer_quantizer = resolve_layer_quantization(
                n,
                bits,
                quantizer,
                mixed_policy,
                num_hidden_layers,
            )
            adaptive_metrics = None
            metric_text = ""
        print(
            f"  [{current_layer}/{total_layers}] {layer_bits}-bit {layer_quantizer}: "
            f"{n} ({w.shape[0]}x{w.shape[1]})"
            + (f" [{metric_text}]" if metric_text else ""),
            end="\r",
        )
        
        if mixed_policy != "adaptive":
            limits_int, scales, zeros, codebooks = quantize_schema(
                w,
                layer_bits,
                layer_quantizer,
                group_size,
                h_diag,
                clip_search,
                codebook_iters,
            )
        outlier_rows, outlier_cols, outlier_values = build_sparse_residual(
            w,
            limits_int,
            scales,
            zeros,
            group_size,
            outlier_fraction=outlier_fraction,
            h_diag=h_diag,
            codebooks=codebooks,
        )
        svd_u, svd_s, svd_v = build_low_rank_residual(
            w,
            limits_int,
            scales,
            zeros,
            group_size,
            outlier_rows=outlier_rows,
            outlier_cols=outlier_cols,
            svd_rank=svd_rank,
            codebooks=codebooks,
        )
        
        # Deploy the TMG-Q Packer manually without float drift
        qlayer = QuantizedLinear(w.shape[1], w.shape[0], bias=m.bias is not None, gs=group_size, n_bits=layer_bits)
        packed_w, shape, pad = pack_nbit(limits_int, layer_bits)
        
        qlayer.qweight = packed_w.cpu()
        qlayer.scales = scales.cpu()
        qlayer.zeros = zeros.cpu()
        qlayer.codebooks = codebooks.cpu()
        qlayer.w_shape = torch.tensor(shape, dtype=torch.int32)
        qlayer.pad_len = torch.tensor(pad, dtype=torch.int32)
        qlayer.n_bits = torch.tensor(layer_bits, dtype=torch.int32)
        qlayer.group_size = torch.tensor(group_size, dtype=torch.int32)
        qlayer.outlier_rows = outlier_rows.cpu()
        qlayer.outlier_cols = outlier_cols.cpu()
        qlayer.outlier_values = outlier_values.cpu()
        qlayer.svd_u = svd_u.cpu()
        qlayer.svd_s = svd_s.cpu()
        qlayer.svd_v = svd_v.cpu()
        if m.bias is not None:
             qlayer.bias.data = m.bias.data.to(torch.float16)
        
        pre, _, post = n.rpartition('.')
        parent = rgetattr(model, pre) if pre else model
        setattr(parent, post, qlayer)
        del m, w, h_diag, limits_int, scales, zeros, codebooks
        del outlier_rows, outlier_cols, outlier_values, svd_u, svd_s, svd_v, packed_w, qlayer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        current_layer += 1

    if quantize_embeddings:
        embedding = model.get_input_embeddings()
        if embedding is None or not isinstance(embedding, nn.Embedding):
            raise RuntimeError("The model does not expose a supported input embedding")
        output = model.get_output_embeddings()
        tied = (
            output is not None
            and hasattr(output, "weight")
            and output.weight.data_ptr() == embedding.weight.data_ptr()
        )
        print(f"\nQuantizing token embedding at {embedding_bits}-bit on {device}...")
        embedding_dtype = embedding.weight.dtype
        weight = embedding.weight.data.to(device, non_blocking=True)
        limits, scales, zeros, codebooks = quantize_schema(
            weight,
            embedding_bits,
            "linear",
            group_size,
            None,
            clip_search,
            codebook_iters,
        )
        packed, shape, pad = pack_nbit(limits, embedding_bits)
        qembedding = QuantizedEmbedding(
            embedding.num_embeddings,
            embedding.embedding_dim,
            gs=group_size,
            n_bits=embedding_bits,
        )
        qembedding.qweight = packed.cpu()
        qembedding.scales = scales.cpu()
        qembedding.zeros = zeros.cpu()
        qembedding.codebooks = codebooks.cpu()
        qembedding.w_shape = torch.tensor(shape, dtype=torch.int32)
        qembedding.pad_len = torch.tensor(pad, dtype=torch.int32)
        qembedding.n_bits = torch.tensor(embedding_bits, dtype=torch.int32)
        qembedding.group_size = torch.tensor(group_size, dtype=torch.int32)
        use_quantized_head = tied and quantize_lm_head
        qembedding.tied_lm_head = torch.tensor(use_quantized_head, dtype=torch.bool)
        dtype_code = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}.get(embedding_dtype, 1)
        qembedding.dtype_code = torch.tensor(dtype_code, dtype=torch.int32)
        if embedding_svd_rank > 0:
            print(f"Building rank-{embedding_svd_rank} randomized vocabulary residual...")
            dequantized = dequantize_schema(limits, scales, zeros, group_size, codebooks=codebooks)
            residual = weight.float() - dequantized
            rank = min(embedding_svd_rank, min(residual.shape))
            q = min(rank + 8, min(residual.shape))
            u, s, v = torch.pca_lowrank(residual, q=q, center=False, niter=2)
            qembedding.svd_u = u[:, :rank].to(torch.float16).cpu()
            qembedding.svd_s = s[:rank].to(torch.float16).cpu()
            qembedding.svd_v = v[:, :rank].t().to(torch.float16).cpu()
            del dequantized, residual, u, s, v
        model.set_input_embeddings(qembedding)
        if use_quantized_head:
            model.set_output_embeddings(QuantizedTiedLMHead(qembedding))
        del embedding, output, embedding_dtype, weight, limits, scales, zeros, codebooks, packed, qembedding
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        
    print(f"\nTMG-Q Packing Complete. Extracting state_dict...")
    
    # Save the completely shrunken architecture
    torch.save(model.state_dict(), export_path)
    print(f"\n=========================================")
    print(f"EXPORT SUCCESSFUL!")
    print(f"Compressed Matrix saved to: {export_path}")
    print(f"You can now send {export_path} to your friend!")
    print(f"=========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export a HuggingFace model with TMG-Q packed weights.")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--bits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-clip-search", action="store_true")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--calib-samples", type=int, default=64)
    parser.add_argument("--calib-length", type=int, default=256)
    parser.add_argument("--outlier-fraction", type=float, default=0.0, help="Sparse residual fraction, e.g. 0.001 for 0.1%%.")
    parser.add_argument("--svd-rank", type=int, default=0, help="Optional low-rank residual rank.")
    parser.add_argument("--quantizer", choices=("linear", "codebook"), default="linear")
    parser.add_argument("--codebook-iters", type=int, default=6)
    parser.add_argument("--mixed-policy", choices=("none", "balanced", "aggressive", "adaptive", "budget"), default="none")
    parser.add_argument("--adaptive-2bit-nmse", type=float, default=0.06)
    parser.add_argument("--adaptive-3bit-nmse", type=float, default=0.02)
    parser.add_argument("--adaptive-probe-rows", type=int, default=256)
    parser.add_argument("--adaptive-min-bits", type=int, choices=(2, 3, 4), default=2)
    parser.add_argument("--quant-device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--target-ratio", type=float, default=3.5)
    parser.add_argument("--quantize-embeddings", action="store_true")
    parser.add_argument("--embedding-bits", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument(
        "--quantize-lm-head",
        action="store_true",
        help="Reuse the quantized token embedding as lm_head. More compression, but much higher quality risk.",
    )
    parser.add_argument("--embedding-svd-rank", type=int, default=0)
    args = parser.parse_args()

    output = args.output or f"{args.model.split('/')[-1]}_{args.bits}bit_TMGQ.pt"
    export_huggingface_model(
        model_name=args.model,
        bits=args.bits,
        export_path=output,
        group_size=args.group_size,
        clip_search=not args.no_clip_search,
        calibrate=args.calibrate,
        calib_samples=args.calib_samples,
        calib_length=args.calib_length,
        outlier_fraction=args.outlier_fraction,
        svd_rank=args.svd_rank,
        quantizer=args.quantizer,
        codebook_iters=args.codebook_iters,
        mixed_policy=args.mixed_policy,
        adaptive_2bit_nmse=args.adaptive_2bit_nmse,
        adaptive_3bit_nmse=args.adaptive_3bit_nmse,
        adaptive_probe_rows=args.adaptive_probe_rows,
        quant_device=args.quant_device,
        target_ratio=args.target_ratio,
        quantize_embeddings=args.quantize_embeddings,
        embedding_bits=args.embedding_bits,
        quantize_lm_head=args.quantize_lm_head,
        embedding_svd_rank=args.embedding_svd_rank,
        adaptive_min_bits=args.adaptive_min_bits,
    )
