import argparse

import torch

from tmgq_transformers_compat import disable_broken_torchvision

disable_broken_torchvision()
from transformers import AutoModelForCausalLM


def tensor_bytes(t):
    return t.numel() * t.element_size()

def unique_state_dict_bytes(state_dict):
    total = 0
    seen = set()
    for tensor in state_dict.values():
        storage = tensor.untyped_storage()
        storage_id = storage.data_ptr()
        if storage_id in seen:
            continue
        seen.add(storage_id)
        total += storage.nbytes()
    return total


def checkpoint_packed_bytes(state_dict):
    total = 0
    for name, tensor in state_dict.items():
        if any(key in name for key in ("qweight", "scales", "zeros", "codebooks", "outlier_rows", "outlier_cols", "outlier_values", "svd_u", "svd_s", "svd_v", "svd_u_scale", "svd_v_scale")):
            total += tensor_bytes(tensor)
    return total

def quantized_prefixes(state_dict):
    return sorted(name.rpartition(".")[0] for name in state_dict if name.endswith(".qweight"))

def checkpoint_payload_bytes_for_prefix(state_dict, prefix):
    total = 0
    for suffix in ("qweight", "scales", "zeros", "codebooks", "outlier_rows", "outlier_cols", "outlier_values", "svd_u", "svd_s", "svd_v", "svd_u_scale", "svd_v_scale"):
        key = f"{prefix}.{suffix}"
        if key in state_dict:
            total += tensor_bytes(state_dict[key])
    return total

def matching_original_fp16_bytes(model, prefixes):
    modules = dict(model.named_modules())
    total = 0
    for prefix in prefixes:
        module = modules.get(prefix)
        if module is not None and (
            isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
            or module.__class__.__name__ == "Conv1D"
        ):
            total += module.weight.numel() * 2
    return total

def bit_distribution(state_dict):
    counts = {}
    weights = {}
    for prefix in quantized_prefixes(state_dict):
        bits_key = f"{prefix}.n_bits"
        shape_key = f"{prefix}.w_shape"
        if bits_key not in state_dict or shape_key not in state_dict:
            continue
        bits = int(state_dict[bits_key].item())
        shape = state_dict[shape_key].tolist()
        num_weights = int(shape[0]) * int(shape[1])
        counts[bits] = counts.get(bits, 0) + 1
        weights[bits] = weights.get(bits, 0) + num_weights
    return counts, weights


def original_quantizable_fp16_bytes(model_name):
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    total = 0
    seen = set()
    for module in model.modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) or module.__class__.__name__ == "Conv1D":
            storage_id = module.weight.data_ptr()
            if storage_id in seen:
                continue
            seen.add(storage_id)
            total += module.weight.numel() * 2
    return total

def original_unique_fp16_bytes(model):
    total = 0
    seen = set()
    for parameter in model.parameters():
        storage_id = parameter.data_ptr()
        if storage_id in seen:
            continue
        seen.add(storage_id)
        total += parameter.numel() * 2
    return total


def main():
    parser = argparse.ArgumentParser(description="Report compressed linear payload size vs FP16 linear baseline.")
    parser.add_argument("--model", required=True)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()

    fp16_linear = original_quantizable_fp16_bytes(args.model)
    print(f"Original quantizable FP16 payload: {fp16_linear / 1024:.2f} KiB")
    for ckpt in args.checkpoints:
        state = torch.load(ckpt, map_location="cpu")
        packed = checkpoint_packed_bytes(state)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16, low_cpu_mem_usage=True)
        original_full = original_unique_fp16_bytes(model)
        checkpoint_full = unique_state_dict_bytes(state)
        checkpoint_file = __import__("pathlib").Path(ckpt).stat().st_size
        prefixes = quantized_prefixes(state)
        matched_fp16 = matching_original_fp16_bytes(model, prefixes)
        matched_packed = sum(checkpoint_payload_bytes_for_prefix(state, prefix) for prefix in prefixes)
        bit_counts, bit_weights = bit_distribution(state)
        print(f"{ckpt}:")
        print(f"  quantized layers: {len(prefixes)}")
        print(f"  matched FP16 payload: {matched_fp16 / 1024:.2f} KiB")
        print(f"  matched packed payload: {matched_packed / 1024:.2f} KiB")
        print(f"  matched ratio: {matched_fp16 / matched_packed:.2f}x")
        print(f"  packed payload in checkpoint: {packed / 1024:.2f} KiB")
        print(f"  full FP16 parameter baseline: {original_full / 1024:.2f} KiB")
        print(f"  full checkpoint tensor payload: {checkpoint_full / 1024:.2f} KiB")
        print(f"  full tensor ratio: {original_full / checkpoint_full:.2f}x")
        print(f"  checkpoint file size: {checkpoint_file / 1024:.2f} KiB")
        print(f"  full file ratio: {original_full / checkpoint_file:.2f}x")
        print(f"  bit layer counts: {dict(sorted(bit_counts.items()))}")
        print(f"  bit weight counts: {dict(sorted(bit_weights.items()))}")


if __name__ == "__main__":
    main()
