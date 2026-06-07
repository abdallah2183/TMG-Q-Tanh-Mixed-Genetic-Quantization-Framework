import torch
import torch.nn as nn
import torch.nn.functional as F
import math

SUPPORTED_BITS = (2, 3, 4)

def values_per_int32(n_bits):
    if n_bits not in SUPPORTED_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_BITS}, got {n_bits}")
    return 32 // n_bits

def pack_nbit(limits_int, n_bits):
    """
    Packs unsigned quantized values into int32 containers.
    2-bit: 16 values/int32, 3-bit: 10 values/int32, 4-bit: 8 values/int32.
    """
    assert limits_int.dtype == torch.int32, "Input must already be cast to int32"
    vals_per_word = values_per_int32(n_bits)
    max_val = (1 << n_bits) - 1
    if limits_int.numel() and (limits_int.min() < 0 or limits_int.max() > max_val):
        raise ValueError(f"Packed values must be in [0, {max_val}] for {n_bits}-bit")

    original_shape = limits_int.shape
    w_flat = limits_int.flatten()
    pad_len = (vals_per_word - (w_flat.numel() % vals_per_word)) % vals_per_word
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len), value=0)

    w_flat = w_flat.view(-1, vals_per_word)
    packed = torch.zeros(w_flat.shape[0], dtype=torch.int32, device=limits_int.device)
    for i in range(vals_per_word):
        shifted = torch.bitwise_left_shift(w_flat[:, i], i * n_bits)
        packed = torch.bitwise_or(packed, shifted)

    return packed, original_shape, pad_len

def unpack_nbit(packed, original_shape, pad_len, n_bits):
    vals_per_word = values_per_int32(n_bits)
    mask = (1 << n_bits) - 1
    unpacked = torch.empty((packed.shape[0], vals_per_word), dtype=torch.int32, device=packed.device)
    for i in range(vals_per_word):
        shifted = torch.bitwise_right_shift(packed, i * n_bits)
        unpacked[:, i] = torch.bitwise_and(shifted, mask)

    w_flat = unpacked.flatten()
    if pad_len > 0:
        w_flat = w_flat[:-pad_len]

    return w_flat.view(original_shape)

def pack_2bit(limits_int):
    return pack_nbit(limits_int, 2)

def unpack_2bit(packed, original_shape, pad_len):
    return unpack_nbit(packed, original_shape, pad_len, 2)

def pack_3bit(limits_int):
    """
    Packs an INT32 tensor containing values [0, 7] down by a factor of 10.
    10 * 3-bit weights = 30 bits, comfortably fitting inside one 32-bit integer box.
    Returns the physically shrunk INT32 tensor.
    """
    return pack_nbit(limits_int, 3)

def unpack_3bit(packed, original_shape, pad_len):
    """
    Extracts the integers from the packed 32-bit buckets instantly onto the GPU.
    """
    return unpack_nbit(packed, original_shape, pad_len, 3)

def pack_4bit(limits_int):
    return pack_nbit(limits_int, 4)

def unpack_4bit(packed, original_shape, pad_len):
    return unpack_nbit(packed, original_shape, pad_len, 4)

def extract_packed_schema(w_q, n_bits=3, gs=128):
    """
    Takes a mathematical quantized Float schema and strictly extracts its INT format,
    scale factor, and zero point offsets per block.
    """
    q_levels = (2**n_bits) - 1
    rows, cols = w_q.shape
    num_blocks = math.ceil(cols / gs)
    
    limits = torch.zeros_like(w_q, dtype=torch.int32)
    scales = torch.zeros((rows, num_blocks), dtype=torch.float16, device=w_q.device)
    zeros = torch.zeros((rows, num_blocks), dtype=torch.float16, device=w_q.device)
    
    for i, cs in enumerate(range(0, cols, gs)):
        ce = min(cs+gs, cols)
        block = w_q[:, cs:ce]
        b_min = block.min(dim=1, keepdim=True).values
        b_max = block.max(dim=1, keepdim=True).values
        scale = (b_max - b_min).clamp(min=1e-8) / q_levels
        
        ws = (block - b_min) / scale
        wr = torch.clamp(torch.round(ws), 0, q_levels)
        
        limits[:, cs:ce] = wr.to(torch.int32)
        scales[:, i:i+1] = scale.to(torch.float16)
        zeros[:, i:i+1] = b_min.to(torch.float16)

    return limits, scales, zeros

class QuantizedLinear(nn.Module):
    """
    A true hardware-packed pseudo-linear layer.
    Holds only INT32 packed weights in VRAM (Shrinkage: 10x per INT32 versus FP16).
    Dynamically unpacks and scales during the Forward pass.
    """
    def __init__(self, in_features, out_features, bias=True, gs=128, n_bits=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gs = gs
        self.n_bits_val = n_bits
        
        # We explicitly use buffers because these are not trainable via autograd
        self.register_buffer('group_size', torch.tensor(gs, dtype=torch.int32))
        self.register_buffer('qweight', torch.empty(0, dtype=torch.int32))
        self.register_buffer('scales', torch.empty(0, dtype=torch.float16))
        self.register_buffer('zeros', torch.empty(0, dtype=torch.float16))
        self.register_buffer('codebooks', torch.empty(0, dtype=torch.float16))
        self.register_buffer('w_shape', torch.empty(2, dtype=torch.int32))
        self.register_buffer('pad_len', torch.tensor(0, dtype=torch.int32))
        self.register_buffer('n_bits', torch.tensor(n_bits, dtype=torch.int32))
        self.register_buffer('outlier_rows', torch.empty(0, dtype=torch.int32))
        self.register_buffer('outlier_cols', torch.empty(0, dtype=torch.int32))
        self.register_buffer('outlier_values', torch.empty(0, dtype=torch.float16))
        self.register_buffer('svd_u', torch.empty(0, dtype=torch.float16))
        self.register_buffer('svd_s', torch.empty(0, dtype=torch.float16))
        self.register_buffer('svd_v', torch.empty(0, dtype=torch.float16))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
    def pack_from_float(self, math_wq, bias_val=None, n_bits=3, outlier_fraction=0.0, h_diag=None, svd_rank=0):
        """
        Called offline during quantization. Accepts mathematical matrix, packs it natively.
        """
        limits_int, scales, zeros = extract_packed_schema(math_wq, n_bits=n_bits, gs=self.gs)
        packed_w, shape, pad = pack_nbit(limits_int, n_bits)
        
        self.qweight = packed_w
        self.scales = scales
        self.zeros = zeros
        self.codebooks = torch.empty(0, dtype=torch.float16, device=math_wq.device)
        self.w_shape = torch.tensor(shape, dtype=torch.int32)
        self.pad_len = torch.tensor(pad, dtype=torch.int32)
        self.n_bits = torch.tensor(n_bits, dtype=torch.int32)
        self.group_size = torch.tensor(self.gs, dtype=torch.int32)

        self.outlier_rows = torch.empty(0, dtype=torch.int32, device=math_wq.device)
        self.outlier_cols = torch.empty(0, dtype=torch.int32, device=math_wq.device)
        self.outlier_values = torch.empty(0, dtype=torch.float16, device=math_wq.device)
        if outlier_fraction > 0:
            rows, cols = math_wq.shape
            cols_idx = torch.arange(cols, device=math_wq.device) // self.gs
            deq = limits_int.float() * scales[:, cols_idx].float() + zeros[:, cols_idx].float()
            residual = math_wq.float() - deq
            score = residual.abs()
            if h_diag is not None and h_diag.numel() >= cols:
                score = score * h_diag[:cols].float().sqrt().view(1, -1).to(score.device)
            k = min(score.numel(), max(1, int(score.numel() * outlier_fraction)))
            flat_idx = torch.topk(score.flatten(), k=k, largest=True).indices
            self.outlier_rows = (flat_idx // cols).to(torch.int32)
            self.outlier_cols = (flat_idx % cols).to(torch.int32)
            self.outlier_values = residual.flatten()[flat_idx].to(torch.float16)

        self.svd_u = torch.empty(0, dtype=torch.float16, device=math_wq.device)
        self.svd_s = torch.empty(0, dtype=torch.float16, device=math_wq.device)
        self.svd_v = torch.empty(0, dtype=torch.float16, device=math_wq.device)
        if svd_rank > 0:
            rows, cols = math_wq.shape
            cols_idx = torch.arange(cols, device=math_wq.device) // self.gs
            deq = limits_int.float() * scales[:, cols_idx].float() + zeros[:, cols_idx].float()
            residual = math_wq.float() - deq
            if self.outlier_values.numel() > 0:
                r = self.outlier_rows.to(dtype=torch.long)
                c = self.outlier_cols.to(dtype=torch.long)
                residual[r, c] = 0.0
            rank = min(svd_rank, min(rows, cols))
            u, s, vh = torch.linalg.svd(residual.cpu(), full_matrices=False)
            self.svd_u = u[:, :rank].to(torch.float16)
            self.svd_s = s[:rank].to(torch.float16)
            self.svd_v = vh[:rank, :].to(torch.float16)
        
        if self.bias is not None and bias_val is not None:
            self.bias.data = bias_val.to(torch.float16)
            
    def forward(self, x):
        device = self.qweight.device
        dtype = x.dtype # likely float32 or float16
        
        # 1. Unpack limits securely back to Int
        shape = tuple(self.w_shape.tolist())
        bits = self.n_bits.item() if hasattr(self, 'n_bits') else self.n_bits_val
        gs = int(self.group_size.item()) if hasattr(self, 'group_size') else self.gs
        
        limits_int = unpack_nbit(self.qweight, shape, self.pad_len.item(), bits)
            
        limits_float = limits_int.to(dtype)
        
        # 2. Dequantize by either linear scale/zero or learned group codebooks.
        cols = limits_float.shape[1]
        group_ids = torch.arange(cols, device=limits_float.device) // gs
        if self.codebooks.numel() > 0:
            codebooks = self.codebooks.to(device=limits_float.device, dtype=dtype)
            group_codebooks = codebooks[:, group_ids, :]
            w_dequantized = torch.gather(group_codebooks, 2, limits_int.to(torch.long).unsqueeze(-1)).squeeze(-1)
        else:
            scales = self.scales[:, group_ids].to(dtype)
            zeros = self.zeros[:, group_ids].to(dtype)
            w_dequantized = (limits_float * scales) + zeros
        if self.outlier_values.numel() > 0:
            rows = self.outlier_rows.to(device=limits_float.device, dtype=torch.long)
            cols = self.outlier_cols.to(device=limits_float.device, dtype=torch.long)
            vals = self.outlier_values.to(device=limits_float.device, dtype=dtype)
            w_dequantized = w_dequantized.clone()
            w_dequantized[rows, cols] = w_dequantized[rows, cols] + vals
        if self.svd_s.numel() > 0:
            u = self.svd_u.to(device=limits_float.device, dtype=dtype)
            s = self.svd_s.to(device=limits_float.device, dtype=dtype)
            v = self.svd_v.to(device=limits_float.device, dtype=dtype)
            w_dequantized = w_dequantized + ((u * s.unsqueeze(0)) @ v)
            
        # 3. Standard Matrix Multiplication Execution
        bias = self.bias.to(dtype) if self.bias is not None else None
        return F.linear(x, w_dequantized, bias)


class QuantizedEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, gs=128, n_bits=4):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.gs = gs
        self.n_bits_val = n_bits

        self.register_buffer("group_size", torch.tensor(gs, dtype=torch.int32))
        self.register_buffer("qweight", torch.empty(0, dtype=torch.int32))
        self.register_buffer("scales", torch.empty(0, dtype=torch.float16))
        self.register_buffer("zeros", torch.empty(0, dtype=torch.float16))
        self.register_buffer("codebooks", torch.empty(0, dtype=torch.float16))
        self.register_buffer("w_shape", torch.empty(2, dtype=torch.int32))
        self.register_buffer("pad_len", torch.tensor(0, dtype=torch.int32))
        self.register_buffer("n_bits", torch.tensor(n_bits, dtype=torch.int32))
        self.register_buffer("tied_lm_head", torch.tensor(False, dtype=torch.bool))
        self.register_buffer("dtype_code", torch.tensor(1, dtype=torch.int32))
        self.register_buffer("svd_u", torch.empty(0, dtype=torch.float16))
        self.register_buffer("svd_s", torch.empty(0, dtype=torch.float16))
        self.register_buffer("svd_v", torch.empty(0, dtype=torch.float16))
        self.register_buffer("svd_u_scale", torch.empty(0, dtype=torch.float16))
        self.register_buffer("svd_v_scale", torch.empty(0, dtype=torch.float16))

    def output_dtype(self):
        code = int(self.dtype_code.item())
        return {0: torch.float16, 1: torch.bfloat16, 2: torch.float32}.get(code, torch.bfloat16)

    def dequantize(self, dtype):
        shape = tuple(self.w_shape.tolist())
        bits = int(self.n_bits.item())
        gs = int(self.group_size.item())
        limits_int = unpack_nbit(self.qweight, shape, self.pad_len.item(), bits)
        group_ids = torch.arange(shape[1], device=limits_int.device) // gs
        if self.codebooks.numel() > 0:
            codebooks = self.codebooks.to(dtype=dtype)
            grouped = codebooks[:, group_ids, :]
            weight = torch.gather(grouped, 2, limits_int.long().unsqueeze(-1)).squeeze(-1)
        else:
            weight = (
                limits_int.to(dtype) * self.scales[:, group_ids].to(dtype)
                + self.zeros[:, group_ids].to(dtype)
            )
        if self.svd_s.numel() > 0:
            u = self.svd_u.to(dtype=dtype)
            s = self.svd_s.to(dtype=dtype)
            v = self.svd_v.to(dtype=dtype)
            if self.svd_u_scale.numel() > 0:
                u_scale = self.svd_u_scale.to(dtype=dtype)
                if u_scale.numel() == u.shape[0]:
                    u = u * u_scale.view(-1, 1)
                else:
                    u = u * u_scale.view(1, -1)
            if self.svd_v_scale.numel() > 0:
                v = v * self.svd_v_scale.to(dtype=dtype).view(-1, 1)
            weight = weight + ((u * s.unsqueeze(0)) @ v)
        return weight

    def forward(self, input_ids):
        return F.embedding(input_ids, self.dequantize(self.output_dtype()))


class QuantizedTiedLMHead(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        object.__setattr__(self, "_embedding_source", embedding)

    def forward(self, x):
        source = object.__getattribute__(self, "_embedding_source")
        return F.linear(x, source.dequantize(x.dtype))


class DistilledQuantizedEmbedding(nn.Module):
    def __init__(self, frozen_embedding, rank):
        super().__init__()
        self.frozen_embedding = frozen_embedding
        self.num_embeddings = frozen_embedding.num_embeddings
        self.embedding_dim = frozen_embedding.embedding_dim
        self.residual_left = nn.Parameter(
            torch.zeros(self.num_embeddings, rank, dtype=torch.float32)
        )
        self.residual_right = nn.Parameter(
            torch.zeros(rank, self.embedding_dim, dtype=torch.float32)
        )

    def initialize_from_residual(self, residual):
        rank = self.residual_left.shape[1]
        q = min(rank + 8, min(residual.shape))
        u, s, v = torch.pca_lowrank(residual.float(), q=q, center=False, niter=2)
        root = s[:rank].sqrt()
        self.residual_left.data.copy_(u[:, :rank] * root.unsqueeze(0))
        self.residual_right.data.copy_(root.unsqueeze(1) * v[:, :rank].t())

    def forward(self, input_ids):
        base = self.frozen_embedding(input_ids)
        left = F.embedding(input_ids, self.residual_left)
        correction = left @ self.residual_right
        return base + correction.to(base.dtype)


class DistilledTiedLMHead(nn.Module):
    def __init__(self, embedding):
        super().__init__()
        object.__setattr__(self, "_embedding_source", embedding)

    def forward(self, x):
        source = object.__getattribute__(self, "_embedding_source")
        base = F.linear(x, source.frozen_embedding.dequantize(x.dtype))
        projected = x.float() @ source.residual_right.t()
        correction = projected @ source.residual_left.t()
        return base + correction.to(base.dtype)
