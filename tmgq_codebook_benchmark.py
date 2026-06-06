import torch
import torch.nn.functional as F

from tmgq_export_llama import codebook_quantize_packable, sensitivity_quantize_packable
from tmgq_packer import QuantizedLinear, pack_nbit


def bytes_of(tensor):
    return tensor.numel() * tensor.element_size()


def make_layer_from_schema(weight, bias, bits, group_size, limits, scales=None, zeros=None, codebooks=None):
    layer = QuantizedLinear(weight.shape[1], weight.shape[0], bias=True, gs=group_size, n_bits=bits)
    packed, shape, pad = pack_nbit(limits, bits)
    layer.qweight = packed
    layer.w_shape = torch.tensor(shape, dtype=torch.int32)
    layer.pad_len = torch.tensor(pad, dtype=torch.int32)
    layer.n_bits = torch.tensor(bits, dtype=torch.int32)
    layer.group_size = torch.tensor(group_size, dtype=torch.int32)
    layer.bias.data = bias.to(torch.float16)
    if codebooks is not None:
        layer.codebooks = codebooks
        layer.scales = torch.empty(0, dtype=torch.float16)
        layer.zeros = torch.empty(0, dtype=torch.float16)
    else:
        layer.scales = scales
        layer.zeros = zeros
    return layer


def payload_bytes(layer):
    return (
        bytes_of(layer.qweight)
        + bytes_of(layer.scales)
        + bytes_of(layer.zeros)
        + bytes_of(layer.codebooks)
    )


def run(bits, inject_outliers=False, group_size=128):
    torch.manual_seed(9000 + bits)
    weight = torch.randn(1024, 1024) * 0.35
    if inject_outliers:
        mask = torch.rand_like(weight) < 0.01
        weight = torch.where(mask, weight * 12.0, weight)
    bias = torch.randn(1024) * 0.01
    x = torch.randn(16, 1024)
    ref = F.linear(x, weight, bias)

    lin_limits, lin_scales, lin_zeros = sensitivity_quantize_packable(weight, bits, gs=group_size, clip_search=True)
    lin_layer = make_layer_from_schema(weight, bias, bits, group_size, lin_limits, lin_scales, lin_zeros)
    lin_out = lin_layer(x).float()

    cb_limits, cb_codebooks = codebook_quantize_packable(weight, bits, gs=group_size, codebook_iters=6)
    cb_layer = make_layer_from_schema(weight, bias, bits, group_size, cb_limits, codebooks=cb_codebooks)
    cb_out = cb_layer(x).float()

    fp16_bytes = weight.numel() * 2
    return {
        "bits": bits,
        "linear_mse": torch.mean((ref - lin_out) ** 2).item(),
        "linear_cos": F.cosine_similarity(ref.flatten(), lin_out.flatten(), dim=0).item(),
        "linear_ratio": fp16_bytes / payload_bytes(lin_layer),
        "codebook_mse": torch.mean((ref - cb_out) ** 2).item(),
        "codebook_cos": F.cosine_similarity(ref.flatten(), cb_out.flatten(), dim=0).item(),
        "codebook_ratio": fp16_bytes / payload_bytes(cb_layer),
    }


def main():
    for inject_outliers in (False, True):
        print("Case:", "1% strong synthetic outliers" if inject_outliers else "normal weights")
        for bits in (2, 3, 4):
            r = run(bits, inject_outliers=inject_outliers)
            print(
                f"{bits}-bit | "
                f"linear cos={r['linear_cos']:.6f} mse={r['linear_mse']:.6f} ratio={r['linear_ratio']:.2f}x | "
                f"codebook cos={r['codebook_cos']:.6f} mse={r['codebook_mse']:.6f} ratio={r['codebook_ratio']:.2f}x"
            )
        print()


if __name__ == "__main__":
    main()
