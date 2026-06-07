import torch
import torch.nn.functional as F

from tmgq_packer import QuantizedLinear


def bytes_of(tensor):
    return tensor.numel() * tensor.element_size()


def benchmark(bits, in_features=1024, out_features=1024, batch=16, group_size=128, outlier_fraction=0.0, inject_outliers=False, svd_rank=0):
    torch.manual_seed(2026 + bits)
    weight = torch.randn(out_features, in_features) * 0.35
    if inject_outliers:
        mask = torch.rand_like(weight) < 0.01
        weight = torch.where(mask, weight * 12.0, weight)
    bias = torch.randn(out_features) * 0.01
    x = torch.randn(batch, in_features)

    reference = F.linear(x, weight, bias)

    qlayer = QuantizedLinear(in_features, out_features, bias=True, gs=group_size, n_bits=bits)
    qlayer.pack_from_float(weight, bias_val=bias, n_bits=bits, outlier_fraction=outlier_fraction, svd_rank=svd_rank)
    actual = qlayer(x)

    fp16_weight_bytes = weight.numel() * 2
    packed_bytes = (
        bytes_of(qlayer.qweight)
        + bytes_of(qlayer.scales)
        + bytes_of(qlayer.zeros)
        + bytes_of(qlayer.outlier_rows)
        + bytes_of(qlayer.outlier_cols)
        + bytes_of(qlayer.outlier_values)
        + bytes_of(qlayer.svd_u)
        + bytes_of(qlayer.svd_s)
        + bytes_of(qlayer.svd_v)
    )
    mse = torch.mean((reference - actual.float()) ** 2).item()
    cosine = F.cosine_similarity(reference.flatten(), actual.float().flatten(), dim=0).item()

    return {
        "bits": bits,
        "fp16_weight_bytes": fp16_weight_bytes,
        "packed_bytes": packed_bytes,
        "ratio": fp16_weight_bytes / packed_bytes,
        "mse": mse,
        "cosine": cosine,
        "qweight_shape": tuple(qlayer.qweight.shape),
        "outliers": qlayer.outlier_values.numel(),
        "svd_rank": qlayer.svd_s.numel(),
    }


def main():
    print("TMG-Q packed Linear layer benchmark")
    print("Layer: 1024 x 1024, group_size=128. FP16 weight size excludes bias, same as ratio denominator baseline.")
    print()
    for inject_outliers in (False, True):
        print("Case:", "1% strong synthetic outliers" if inject_outliers else "normal weights")
        for outlier_fraction, svd_rank in ((0.0, 0), (0.01, 0), (0.01, 4), (0.01, 8)):
            print(f"Sparse residual fraction: {outlier_fraction:.3%}, SVD rank: {svd_rank}")
            for bits in (2, 3, 4):
                r = benchmark(bits, outlier_fraction=outlier_fraction, inject_outliers=inject_outliers, svd_rank=svd_rank)
                print(
                    f"  {bits}-bit | fp16={r['fp16_weight_bytes'] / 1024:.1f} KiB "
                    f"packed={r['packed_bytes'] / 1024:.1f} KiB "
                    f"ratio={r['ratio']:.2f}x "
                    f"mse={r['mse']:.6f} cosine={r['cosine']:.6f} "
                    f"outliers={r['outliers']} svd={r['svd_rank']}"
                )
            print()


if __name__ == "__main__":
    main()
