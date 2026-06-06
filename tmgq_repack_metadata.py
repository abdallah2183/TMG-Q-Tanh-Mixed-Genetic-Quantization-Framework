import argparse
from pathlib import Path

import torch


def convert_matching(state, suffixes, dtype):
    converted = []
    for name, tensor in list(state.items()):
        if not any(name.endswith(suffix) for suffix in suffixes):
            continue
        if not tensor.is_floating_point() or tensor.numel() == 0:
            continue
        state[name] = tensor.to(dtype)
        converted.append(name)
    return converted

def quantize_residual_int8(state):
    converted = []
    for name, tensor in list(state.items()):
        if name.endswith(".svd_u") and tensor.numel() > 0:
            prefix = name.removesuffix(".svd_u")
            if f"{prefix}.tied_lm_head" not in state:
                continue
            scale = tensor.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0
            state[name] = torch.round(tensor.float() / scale.view(-1, 1)).clamp(-127, 127).to(torch.int8)
            state[name.removesuffix("svd_u") + "svd_u_scale"] = scale.to(torch.float16)
            converted.append(name)
        elif name.endswith(".svd_v") and tensor.numel() > 0:
            prefix = name.removesuffix(".svd_v")
            if f"{prefix}.tied_lm_head" not in state:
                continue
            scale = tensor.float().abs().amax(dim=1).clamp(min=1e-8) / 127.0
            state[name] = torch.round(tensor.float() / scale.view(-1, 1)).clamp(-127, 127).to(torch.int8)
            state[name.removesuffix("svd_v") + "svd_v_scale"] = scale.to(torch.float16)
            converted.append(name)
    return converted


def main():
    parser = argparse.ArgumentParser(description="Repack TMG-Q floating metadata with a smaller dtype.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--components",
        choices=("residual", "codebooks", "linear", "all"),
        default="residual",
    )
    parser.add_argument(
        "--dtype",
        choices=("float8_e4m3fn", "float8_e5m2", "int8_symmetric"),
        default="float8_e4m3fn",
    )
    args = parser.parse_args()

    dtype = getattr(torch, args.dtype) if args.dtype != "int8_symmetric" else None
    suffix_groups = {
        "residual": (".svd_u", ".svd_v"),
        "codebooks": (".codebooks",),
        "linear": (".scales", ".zeros"),
        "all": (".svd_u", ".svd_v", ".codebooks", ".scales", ".zeros"),
    }
    state = torch.load(args.checkpoint, map_location="cpu")
    before = sum(t.untyped_storage().nbytes() for t in state.values())
    if args.dtype == "int8_symmetric":
        if args.components not in ("residual", "all"):
            raise ValueError("int8_symmetric currently supports residual or all components")
        converted = quantize_residual_int8(state)
        if args.components == "all":
            converted.extend(convert_matching(state, (".codebooks", ".scales", ".zeros"), torch.float8_e4m3fn))
    else:
        converted = convert_matching(state, suffix_groups[args.components], dtype)
    after = sum(t.untyped_storage().nbytes() for t in state.values())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, args.output)
    print(f"Converted tensors: {len(converted)}")
    print(f"Tensor payload: {before / 1024**2:.2f} MiB -> {after / 1024**2:.2f} MiB")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
