import torch

from tmgq_export_llama import (
    adaptive_quantize_schema,
    adaptive_quantize_with_probe,
    codebook_quantize_packable,
    resolve_layer_quantization,
)
from tmgq_packer import (
    DistilledQuantizedEmbedding,
    DistilledTiedLMHead,
    QuantizedEmbedding,
    QuantizedLinear,
    QuantizedTiedLMHead,
    pack_nbit,
    unpack_nbit,
)


def test_pack_unpack_roundtrip():
    torch.manual_seed(0)
    for bits in (2, 3, 4):
        max_val = (1 << bits) - 1
        values = torch.randint(0, max_val + 1, (17, 19), dtype=torch.int32)
        packed, shape, pad_len = pack_nbit(values, bits)
        restored = unpack_nbit(packed, shape, pad_len, bits)
        assert torch.equal(values, restored)


def test_quantized_linear_forward_shape_all_bits():
    torch.manual_seed(1)
    weight = torch.randn(7, 11)
    x = torch.randn(3, 11)

    for bits in (2, 3, 4):
        layer = QuantizedLinear(11, 7, bias=False, gs=4, n_bits=bits)
        layer.pack_from_float(weight, n_bits=bits)
        y = layer(x)
        assert y.shape == (3, 7)
        assert torch.isfinite(y).all()


def test_codebook_forward_all_bits():
    torch.manual_seed(2)
    weight = torch.randn(7, 11)
    x = torch.randn(3, 11)

    for bits in (2, 3, 4):
        limits, codebooks = codebook_quantize_packable(weight, bits, gs=4, codebook_iters=3)
        packed, shape, pad_len = pack_nbit(limits, bits)
        layer = QuantizedLinear(11, 7, bias=False, gs=4, n_bits=bits)
        layer.qweight = packed
        layer.w_shape = torch.tensor(shape, dtype=torch.int32)
        layer.pad_len = torch.tensor(pad_len, dtype=torch.int32)
        layer.n_bits = torch.tensor(bits, dtype=torch.int32)
        layer.group_size = torch.tensor(4, dtype=torch.int32)
        layer.scales = torch.empty(0, dtype=torch.float16)
        layer.zeros = torch.empty(0, dtype=torch.float16)
        layer.codebooks = codebooks

        y = layer(x)
        assert limits.min() >= 0
        assert limits.max() < (1 << bits)
        assert codebooks.shape == (7, 3, 1 << bits)
        assert y.shape == (3, 7)
        assert torch.isfinite(y).all()


def test_mixed_precision_policies():
    assert resolve_layer_quantization("model.layers.0.mlp.up_proj", 4, "linear", "balanced", 22) == (4, "linear")
    assert resolve_layer_quantization("model.layers.10.mlp.up_proj", 4, "linear", "balanced", 22) == (3, "codebook")
    assert resolve_layer_quantization("model.layers.10.mlp.up_proj", 4, "linear", "aggressive", 22) == (2, "codebook")
    assert resolve_layer_quantization("model.layers.10.self_attn.q_proj", 4, "linear", "aggressive", 22) == (3, "codebook")


def test_adaptive_quantization_thresholds():
    torch.manual_seed(3)
    weight = torch.randn(8, 16)

    choice_2 = adaptive_quantize_schema(weight, 8, None, True, 2, 1.0, 1.0)
    assert choice_2[0:2] == (2, "codebook")

    choice_3 = adaptive_quantize_schema(weight, 8, None, True, 2, 0.0, 1.0)
    assert choice_3[0:2] == (3, "codebook")

    choice_4 = adaptive_quantize_schema(weight, 8, None, True, 2, 0.0, 0.0)
    best_key = min(choice_4[6], key=choice_4[6].get)
    assert choice_4[0] == int(best_key[0])

    choice_min_3 = adaptive_quantize_schema(weight, 8, None, True, 2, 1.0, 1.0, minimum_bits=3)
    assert choice_min_3[0] >= 3


def test_adaptive_probe_quantizes_full_weight():
    torch.manual_seed(4)
    weight = torch.randn(19, 16)
    choice = adaptive_quantize_with_probe(weight, 8, None, True, 2, 1.0, 1.0, probe_rows=5)

    assert choice[0:2] == (2, "codebook")
    assert choice[2].shape == weight.shape
    assert choice[5].shape[0] == weight.shape[0]


def test_quantized_embedding_and_tied_head():
    torch.manual_seed(5)
    weight = torch.randn(23, 12)
    limits, codebooks = codebook_quantize_packable(weight, 3, gs=4, codebook_iters=2)
    packed, shape, pad_len = pack_nbit(limits, 3)

    embedding = QuantizedEmbedding(23, 12, gs=4, n_bits=3)
    embedding.qweight = packed
    embedding.w_shape = torch.tensor(shape, dtype=torch.int32)
    embedding.pad_len = torch.tensor(pad_len, dtype=torch.int32)
    embedding.n_bits = torch.tensor(3, dtype=torch.int32)
    embedding.group_size = torch.tensor(4, dtype=torch.int32)
    embedding.scales = torch.empty(0, dtype=torch.float16)
    embedding.zeros = torch.empty(0, dtype=torch.float16)
    embedding.codebooks = codebooks
    embedding.tied_lm_head = torch.tensor(True)

    input_ids = torch.tensor([[1, 4, 9]])
    hidden = embedding(input_ids)
    logits = QuantizedTiedLMHead(embedding)(hidden)
    assert hidden.shape == (1, 3, 12)
    assert logits.shape == (1, 3, 23)
    assert torch.isfinite(hidden).all()
    assert torch.isfinite(logits).all()


def test_distilled_vocabulary_receives_gradients():
    torch.manual_seed(6)
    base = QuantizedEmbedding(13, 8, gs=4, n_bits=4)
    weight = torch.randn(13, 8)
    base.qweight, shape, pad_len = pack_nbit(
        torch.randint(0, 16, weight.shape, dtype=torch.int32),
        4,
    )
    base.w_shape = torch.tensor(shape, dtype=torch.int32)
    base.pad_len = torch.tensor(pad_len, dtype=torch.int32)
    base.scales = torch.ones(13, 2, dtype=torch.float16)
    base.zeros = torch.zeros(13, 2, dtype=torch.float16)
    distilled = DistilledQuantizedEmbedding(base, rank=3)

    hidden = distilled(torch.tensor([[1, 2, 3]]))
    logits = DistilledTiedLMHead(distilled)(hidden)
    logits.square().mean().backward()
    assert distilled.residual_left.grad is not None
    assert distilled.residual_right.grad is not None


def test_quantized_embedding_int8_residual():
    torch.manual_seed(7)
    embedding = QuantizedEmbedding(9, 6, gs=3, n_bits=4)
    indices = torch.randint(0, 16, (9, 6), dtype=torch.int32)
    embedding.qweight, shape, pad_len = pack_nbit(indices, 4)
    embedding.w_shape = torch.tensor(shape, dtype=torch.int32)
    embedding.pad_len = torch.tensor(pad_len, dtype=torch.int32)
    embedding.scales = torch.ones(9, 2, dtype=torch.float16)
    embedding.zeros = torch.zeros(9, 2, dtype=torch.float16)
    embedding.svd_u = torch.randint(-127, 128, (9, 2), dtype=torch.int8)
    embedding.svd_s = torch.ones(2, dtype=torch.float16)
    embedding.svd_v = torch.randint(-127, 128, (2, 6), dtype=torch.int8)
    embedding.svd_u_scale = torch.full((9,), 0.01, dtype=torch.float16)
    embedding.svd_v_scale = torch.full((2,), 0.01, dtype=torch.float16)

    output = embedding(torch.tensor([[0, 3, 8]]))
    assert output.shape == (1, 3, 6)
    assert torch.isfinite(output).all()
