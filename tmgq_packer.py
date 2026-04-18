import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def pack_3bit(limits_int):
    """
    Packs an INT32 tensor containing values [0, 7] down by a factor of 10.
    10 * 3-bit weights = 30 bits, comfortably fitting inside one 32-bit integer box.
    Returns the physically shrunk INT32 tensor.
    """
    assert limits_int.dtype == torch.int32, "Input must already be cast to int32"
    
    device = limits_int.device
    original_shape = limits_int.shape
    
    w_flat = limits_int.flatten()
    n = w_flat.numel()
    
    # Pad to multiple of 10
    pad_len = (10 - (n % 10)) % 10
    if pad_len > 0:
        w_flat = F.pad(w_flat, (0, pad_len), value=0)
        
    w_flat = w_flat.view(-1, 10)
    num_packed = w_flat.shape[0]
    
    packed = torch.zeros(num_packed, dtype=torch.int32, device=device)
    
    for i in range(10):
        # Shift each col by i*3 bits and OR it into the packed integer
        shifted = torch.bitwise_left_shift(w_flat[:, i], i * 3)
        packed = torch.bitwise_or(packed, shifted)
        
    return packed, original_shape, pad_len

def unpack_3bit(packed, original_shape, pad_len):
    """
    Extracts the integers from the packed 32-bit buckets instantly onto the GPU.
    """
    device = packed.device
    num_packed = packed.shape[0]
    
    unpacked = torch.zeros((num_packed, 10), dtype=torch.int32, device=device)
    
    for i in range(10):
        # Shift right and isolate the bottom 3 bits (0b111 = 7)
        shifted = torch.bitwise_right_shift(packed, i * 3)
        extracted = torch.bitwise_and(shifted, 7)
        unpacked[:, i] = extracted
        
    w_flat = unpacked.flatten()
    
    if pad_len > 0:
        # Trim the padding
        w_flat = w_flat[:-pad_len]
        
    return w_flat.view(original_shape)

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
    def __init__(self, in_features, out_features, bias=True, gs=128):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gs = gs
        
        # We explicitly use buffers because these are not trainable via autograd
        self.register_buffer('qweight', torch.empty(0, dtype=torch.int32))
        self.register_buffer('scales', torch.empty(0, dtype=torch.float16))
        self.register_buffer('zeros', torch.empty(0, dtype=torch.float16))
        self.register_buffer('w_shape', torch.empty(2, dtype=torch.int32))
        self.register_buffer('pad_len', torch.tensor(0, dtype=torch.int32))
        
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float16))
        else:
            self.register_parameter('bias', None)
            
    def pack_from_float(self, math_wq, bias_val=None, n_bits=3):
        """
        Called offline during quantization. Accepts mathematical matrix, packs it natively.
        """
        limits_int, scales, zeros = extract_packed_schema(math_wq, n_bits=n_bits, gs=self.gs)
        packed_w, shape, pad = pack_3bit(limits_int)
        
        self.qweight = packed_w
        self.scales = scales
        self.zeros = zeros
        self.w_shape = torch.tensor(shape, dtype=torch.int32)
        self.pad_len = torch.tensor(pad, dtype=torch.int32)
        
        if self.bias is not None and bias_val is not None:
            self.bias.data = bias_val.to(torch.float16)
            
    def forward(self, x):
        device = self.qweight.device
        dtype = x.dtype # likely float32 or float16
        
        # 1. Unpack limits securely back to Int
        shape = tuple(self.w_shape.tolist())
        limits_int = unpack_3bit(self.qweight, shape, self.pad_len.item())
        limits_float = limits_int.to(dtype)
        
        # 2. Dequantize Asymmetrically by Groups
        # cols = in_features
        w_dequantized = torch.zeros_like(limits_float)
        
        for i, cs in enumerate(range(0, self.in_features, self.gs)):
            ce = min(cs+self.gs, self.in_features)
            s = self.scales[:, i:i+1].to(dtype)
            z = self.zeros[:, i:i+1].to(dtype) # Actually holds b_min directly!
            # Reverse: w = (limits * s) + b_min
            w_dequantized[:, cs:ce] = (limits_float[:, cs:ce] * s) + z
            
        # 3. Standard Matrix Multiplication Execution
        return F.linear(x, w_dequantized, self.bias)
