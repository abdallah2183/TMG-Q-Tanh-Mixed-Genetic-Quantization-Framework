from tmgq_budget_optimizer import (
    QuantizationCandidate,
    estimate_payload_bytes,
    optimize_layer_budget,
)


def test_payload_estimate_orders_bits():
    size_2 = estimate_payload_bytes(512, 2048, 2, "codebook", 64)
    size_3 = estimate_payload_bytes(512, 2048, 3, "codebook", 64)
    size_4 = estimate_payload_bytes(512, 2048, 4, "linear", 64)
    assert size_2 < size_4 < size_3


def test_optimizer_spends_bits_on_sensitive_layer():
    candidates = [
        QuantizationCandidate("sensitive", 2, "codebook", 100.0, 20),
        QuantizationCandidate("sensitive", 4, "linear", 1.0, 40),
        QuantizationCandidate("easy", 2, "codebook", 2.0, 20),
        QuantizationCandidate("easy", 4, "linear", 1.0, 40),
    ]
    plan = optimize_layer_budget(candidates, fp16_bytes=160, target_ratio=2.5)
    assert plan.choices["sensitive"].bits == 4
    assert plan.choices["easy"].bits == 2
    assert plan.payload_bytes == 60


def test_optimizer_rejects_impossible_ratio():
    candidates = [
        QuantizationCandidate("layer", 2, "codebook", 1.0, 30),
        QuantizationCandidate("layer", 4, "linear", 0.1, 50),
    ]
    try:
        optimize_layer_budget(candidates, fp16_bytes=160, target_ratio=30.0)
    except ValueError as exc:
        assert "maximum theoretical ratio" in str(exc)
    else:
        raise AssertionError("Expected an infeasible target to be rejected")
