import torch

from tmgq_export_llama import sensitivity_quantize_packable


def dequantize(limits, scales, zeros, group_size):
    cols = limits.shape[1]
    group_ids = torch.arange(cols, device=limits.device) // group_size
    return limits.float() * scales[:, group_ids].float() + zeros[:, group_ids].float()


def run_case(bits, with_outliers):
    torch.manual_seed(1234)
    group_size = 128
    w = torch.randn(512, 512) * 0.35

    if with_outliers:
        mask = torch.rand_like(w) < 0.01
        w = torch.where(mask, w * 12.0, w)

    h_diag = torch.linspace(0.2, 2.5, w.shape[1])

    q_base, s_base, z_base = sensitivity_quantize_packable(
        w,
        bits,
        gs=group_size,
        clip_search=False,
        h_diag=None,
    )
    q_clip, s_clip, z_clip = sensitivity_quantize_packable(
        w,
        bits,
        gs=group_size,
        clip_search=True,
        h_diag=None,
    )
    q_hess, s_hess, z_hess = sensitivity_quantize_packable(
        w,
        bits,
        gs=group_size,
        clip_search=True,
        h_diag=h_diag,
    )

    d_base = dequantize(q_base, s_base, z_base, group_size)
    d_clip = dequantize(q_clip, s_clip, z_clip, group_size)
    d_hess = dequantize(q_hess, s_hess, z_hess, group_size)

    mse_base = torch.mean((w - d_base) ** 2).item()
    mse_clip = torch.mean((w - d_clip) ** 2).item()
    mse_hess = torch.mean((w - d_hess) ** 2).item()
    weighted_base = torch.mean(((w - d_base) ** 2) * h_diag.view(1, -1)).item()
    weighted_hess = torch.mean(((w - d_hess) ** 2) * h_diag.view(1, -1)).item()

    return {
        "bits": bits,
        "outliers": with_outliers,
        "mse_base": mse_base,
        "mse_clip": mse_clip,
        "mse_hess": mse_hess,
        "clip_gain": (mse_base - mse_clip) / mse_base * 100.0,
        "hess_gain": (weighted_base - weighted_hess) / weighted_base * 100.0,
    }


def main():
    print("TMG-Q quick synthetic benchmark")
    print("Lower MSE is better. Gains are relative to old min/max quantization.")
    print()
    for with_outliers in (False, True):
        print(f"Case: {'with 1% strong outliers' if with_outliers else 'normal weights'}")
        for bits in (2, 3, 4):
            r = run_case(bits, with_outliers)
            print(
                f"  {bits}-bit | old={r['mse_base']:.6f} "
                f"clip={r['mse_clip']:.6f} hess={r['mse_hess']:.6f} "
                f"clip_gain={r['clip_gain']:.2f}% weighted_hess_gain={r['hess_gain']:.2f}%"
            )
        print()


if __name__ == "__main__":
    main()
