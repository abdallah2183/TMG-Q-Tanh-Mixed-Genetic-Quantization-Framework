from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QuantizationCandidate:
    layer: str
    bits: int
    quantizer: str
    loss: float
    payload_bytes: int


@dataclass(frozen=True)
class BudgetPlan:
    choices: dict
    payload_bytes: int
    fp16_bytes: int
    target_bytes: int
    total_loss: float

    @property
    def ratio(self):
        return self.fp16_bytes / max(1, self.payload_bytes)


def estimate_payload_bytes(rows, cols, bits, quantizer, group_size, outlier_fraction=0.0):
    values = rows * cols
    packed = math.ceil(values * bits / 32) * 4
    groups = math.ceil(cols / group_size)

    if quantizer == "codebook":
        metadata = rows * groups * (1 << bits) * 2
    elif quantizer == "linear":
        metadata = rows * groups * 2 * 2
    else:
        raise ValueError(f"Unsupported quantizer: {quantizer}")

    residual_values = max(0, int(values * outlier_fraction))
    sparse_residual = residual_values * (4 + 4 + 2)
    return packed + metadata + sparse_residual


def optimize_layer_budget(candidates, fp16_bytes, target_ratio):
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    if not candidates:
        raise ValueError("No candidates were supplied")

    by_layer = {}
    for candidate in candidates:
        by_layer.setdefault(candidate.layer, []).append(candidate)

    target_bytes = math.floor(fp16_bytes / target_ratio)
    minimum = {layer: min(options, key=lambda item: item.payload_bytes) for layer, options in by_layer.items()}
    minimum_bytes = sum(item.payload_bytes for item in minimum.values())
    maximum_ratio = fp16_bytes / max(1, minimum_bytes)
    if minimum_bytes > target_bytes:
        raise ValueError(
            f"Target {target_ratio:.2f}x is infeasible with the available candidates; "
            f"maximum theoretical ratio is {maximum_ratio:.2f}x"
        )

    def choose(multiplier):
        selected = {}
        for layer, options in by_layer.items():
            selected[layer] = min(
                options,
                key=lambda item: item.loss + multiplier * (item.payload_bytes / fp16_bytes),
            )
        return selected

    low = 0.0
    high = 1.0
    while sum(item.payload_bytes for item in choose(high).values()) > target_bytes:
        high *= 2.0
        if high > 1e18:
            break

    best = minimum
    for _ in range(100):
        mid = (low + high) / 2.0
        selected = choose(mid)
        size = sum(item.payload_bytes for item in selected.values())
        if size <= target_bytes:
            best = selected
            high = mid
        else:
            low = mid

    payload_bytes = sum(item.payload_bytes for item in best.values())
    return BudgetPlan(
        choices=best,
        payload_bytes=payload_bytes,
        fp16_bytes=fp16_bytes,
        target_bytes=target_bytes,
        total_loss=sum(item.loss for item in best.values()),
    )
