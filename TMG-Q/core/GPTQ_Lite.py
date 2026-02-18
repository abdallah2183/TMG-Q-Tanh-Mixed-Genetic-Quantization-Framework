"""
GPTQ-Lite: Simplified high-quality quantization for LLMs
Combines multiple techniques for better results
"""

import numpy as np
import torch

class GPTQLite:
    """
    Advanced quantization combining:
    - Outlier detection & handling
    - Group-wise quantization (128 elements per group)
    - Mixed precision (critical layers stay FP16)
    """
    
    def __init__(self, group_size=128):
        self.group_size = group_size
        self.scales = None
        self.zero_points = None
        self.constants = None
        self.outlier_mask = None
        self.outlier_values = None

    def _detect_outliers_by_percent(self, weights, outlier_percent=3.0):
        if weights.size == 0:
            return np.zeros_like(weights, dtype=bool)

        outlier_percent = float(outlier_percent)
        outlier_percent = max(0.0, min(100.0, outlier_percent))
        if outlier_percent == 0.0:
            return np.zeros_like(weights, dtype=bool)

        threshold = np.percentile(np.abs(weights), 100.0 - outlier_percent)
        return np.abs(weights) >= threshold

    def _paper_compress_raw(self, w, c):
        eps = 1e-12
        w_sign = np.where(w >= 0, 1.0, -1.0)
        unity = w / (w + w_sign * eps)
        denom = np.abs(np.tanh(c)) + (unity - np.abs(c))
        denom = np.where(np.abs(denom) < eps, w_sign * eps, denom)
        return w / denom

    def _paper_decompress(self, q, c):
        return q * (np.tanh(c) - c) + q

    def _calibrate_paper_constant(self, weights, search_range=(0.1, 2.0), steps=20):
        best_c = 0.5
        best_mse = float('inf')

        lo, hi = float(search_range[0]), float(search_range[1])
        if steps is None or int(steps) <= 1:
            candidates = [best_c]
        else:
            candidates = np.linspace(lo, hi, int(steps))

        for c in candidates:
            try:
                raw_q = self._paper_compress_raw(weights, float(c))
                q_int = np.clip(np.round(raw_q), -8, 7).astype(np.int8)
                restored = self._paper_decompress(q_int.astype(np.float32), float(c)).astype(np.float32)
                mse = float(np.mean((weights - restored) ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_c = float(c)
            except Exception:
                continue

        return best_c

    def _pack_int4_signed(self, q_int8):
        q_u = (q_int8.astype(np.int16) + 8).astype(np.uint8)
        if (q_u < 0).any() or (q_u > 15).any():
            q_u = np.clip(q_u, 0, 15).astype(np.uint8)

        if len(q_u) % 2 == 1:
            q_u = np.concatenate([q_u, np.array([8], dtype=np.uint8)])

        packed = np.zeros(len(q_u) // 2, dtype=np.uint8)
        packed |= (q_u[0::2] << 4)
        packed |= (q_u[1::2] & 0x0F)
        return packed

    def _unpack_int4_signed(self, packed, n_values):
        n_values = int(n_values)
        n_elements = len(packed) * 2
        unpacked_u = np.zeros(n_elements, dtype=np.uint8)
        unpacked_u[0::2] = (packed >> 4)
        unpacked_u[1::2] = (packed & 0x0F)
        unpacked_u = unpacked_u[:n_values]
        return (unpacked_u.astype(np.int16) - 8).astype(np.int8)
        
    def detect_outliers(self, weights, threshold=5.0):
        """
        Detect outliers (values > threshold * std)
        These will be stored separately in higher precision
        """
        mean = np.mean(weights)
        std = np.std(weights)
        outlier_threshold = mean + threshold * std
        
        outlier_mask = np.abs(weights) > outlier_threshold
        return outlier_mask
    
    def compress_with_outliers(self, weights, mode="linear"):
        """
        Main compression with outlier handling
        """
        original_shape = weights.shape
        weights_flat = weights.flatten()
        
        # 1. Detect outliers
        if mode == "nonlinear":
            self.outlier_mask = self._detect_outliers_by_percent(weights_flat, outlier_percent=3.0)
        else:
            self.outlier_mask = self.detect_outliers(weights_flat, threshold=4.0)
        n_outliers = np.sum(self.outlier_mask)
        
        print(f"   [Outliers] Found {n_outliers}/{len(weights_flat)} ({n_outliers/len(weights_flat)*100:.2f}%)")
        
        # Store outliers separately (FP16 for space saving)
        if n_outliers > 0:
            self.outlier_values = weights_flat[self.outlier_mask].astype(np.float16)
        else:
            self.outlier_values = np.array([], dtype=np.float16)
        
        # 2. Quantization for non-outlier values
        non_outlier_data = weights_flat[~self.outlier_mask]

        if mode == "nonlinear":
            c = self._calibrate_paper_constant(non_outlier_data)
            self.constants = np.array([c], dtype=np.float32)

            raw_q = self._paper_compress_raw(non_outlier_data.astype(np.float32), c)
            q_int = np.clip(np.round(raw_q), -8, 7).astype(np.int8)

            packed = self._pack_int4_signed(q_int)

            return {
                'packed': packed,
                'scales': None,
                'zero_points': None,
                'constants': self.constants,
                'outlier_mask': self.outlier_mask,
                'outlier_values': self.outlier_values,
                'original_shape': original_shape,
                'n_non_outliers': len(non_outlier_data),
                'type': 'gptq_lite_paper_nonlinear'
            }
        
        # Pad to multiple of group_size
        n_groups = int(np.ceil(len(non_outlier_data) / self.group_size))
        padded_size = n_groups * self.group_size
        padded_data = np.zeros(padded_size, dtype=np.float32)
        padded_data[:len(non_outlier_data)] = non_outlier_data
        
        # Reshape into groups
        grouped = padded_data.reshape(n_groups, self.group_size)
        
        # Quantize each group
        self.scales = np.zeros(n_groups, dtype=np.float32)
        self.zero_points = np.zeros(n_groups, dtype=np.uint8)
        self.constants = None
        quantized_groups = np.zeros_like(grouped, dtype=np.uint8)
        
        for i in range(n_groups):
            group_to_quantize = grouped[i]

            min_val = np.min(group_to_quantize)
            max_val = np.max(group_to_quantize)
            
            scale = (max_val - min_val) / 15.0
            if scale == 0:
                scale = 1.0
            
            zero_point = round(-min_val / scale)
            zero_point = max(0, min(15, zero_point))
            
            self.scales[i] = scale
            self.zero_points[i] = zero_point
            
            # Quantize
            scaled = group_to_quantize / scale + zero_point
            quantized_groups[i] = np.clip(np.round(scaled), 0, 15).astype(np.uint8)
        
        # 3. Bit-pack the quantized data
        quantized_flat = quantized_groups.flatten()
        packed = np.zeros(len(quantized_flat) // 2, dtype=np.uint8)
        packed |= (quantized_flat[0::2] << 4)
        packed |= (quantized_flat[1::2] & 0x0F)
        
        return {
            'packed': packed,
            'scales': self.scales,
            'zero_points': self.zero_points,
            'constants': self.constants,
            'outlier_mask': self.outlier_mask,
            'outlier_values': self.outlier_values,
            'original_shape': original_shape,
            'n_non_outliers': len(non_outlier_data),
            'type': 'gptq_lite_linear'
        }
    
    def decompress_with_outliers(self, compressed_data):
        """
        Decompress with outlier restoration
        """
        packed = compressed_data['packed']
        scales = compressed_data['scales']
        zero_points = compressed_data['zero_points']
        constants = compressed_data.get('constants', None)
        outlier_mask = compressed_data['outlier_mask']
        outlier_values = compressed_data['outlier_values']
        original_shape = compressed_data['original_shape']
        n_non_outliers = compressed_data['n_non_outliers']

        if scales is None or zero_points is None:
            c = 0.5
            if constants is not None and len(constants) > 0:
                c = float(constants[0])

            q_int = self._unpack_int4_signed(packed, n_non_outliers)
            dequantized_flat = self._paper_decompress(q_int.astype(np.float32), c).astype(np.float32)

            total_elements = np.prod(original_shape)
            restored = np.zeros(total_elements, dtype=np.float32)
            restored[~outlier_mask] = dequantized_flat
            restored[outlier_mask] = outlier_values.astype(np.float32)
            return restored.reshape(original_shape)
        
        # 1. Unpack bits
        n_elements = len(packed) * 2
        unpacked = np.zeros(n_elements, dtype=np.float32)
        unpacked[0::2] = (packed >> 4)
        unpacked[1::2] = (packed & 0x0F)
        
        # 2. Dequantize groups
        n_groups = len(scales)
        unpacked_grouped = unpacked.reshape(n_groups, self.group_size)
        dequantized = np.zeros_like(unpacked_grouped, dtype=np.float32)
        
        for i in range(n_groups):
            dequantized[i] = (unpacked_grouped[i] - zero_points[i]) * scales[i]
        
        # Take only valid data (remove padding)
        dequantized_flat = dequantized.flatten()[:n_non_outliers]
        
        # 3. Reconstruct full array with outliers
        total_elements = np.prod(original_shape)
        restored = np.zeros(total_elements, dtype=np.float32)
        
        restored[~outlier_mask] = dequantized_flat
        restored[outlier_mask] = outlier_values.astype(np.float32)
        
        return restored.reshape(original_shape)


def test_gptq_lite():
    """Test on realistic LLM weight distribution"""
    print("="*70)
    print("GPTQ-Lite: Testing on LLM-like weights")
    print("="*70)
    
    # Create realistic distribution with outliers
    np.random.seed(42)
    N = 1_000_000
    
    # Most weights are small
    normal_weights = np.random.normal(0, 0.5, int(N * 0.95))
    # Few outliers
    outliers = np.random.choice([-8, -6, 6, 8], int(N * 0.05))
    
    weights = np.concatenate([normal_weights, outliers])
    np.random.shuffle(weights)
    weights = weights.astype(np.float32)
    
    print(f"\n1. Original weights:")
    print(f"   Size: {weights.nbytes / 1024:.2f} KB")
    print(f"   Mean: {np.mean(weights):.4f}, Std: {np.std(weights):.4f}")
    print(f"   Min: {np.min(weights):.4f}, Max: {np.max(weights):.4f}")
    print(f"   Sample: {weights[:5]}")
    
    print(f"\n2. Compressing with GPTQ-Lite...")
    compressor = GPTQLite(group_size=128)
    compressed = compressor.compress_with_outliers(weights)
    
    # Calculate total size
    total_size = (
        compressed['packed'].nbytes +
        compressed['scales'].nbytes +
        compressed['zero_points'].nbytes +
        compressed['outlier_mask'].nbytes +
        compressed['outlier_values'].nbytes
    )
    
    print(f"\n3. Compression result:")
    print(f"   Compressed size: {total_size / 1024:.2f} KB")
    print(f"   Ratio: {weights.nbytes / total_size:.2f}x")
    
    print(f"\n4. Decompressing...")
    restored = compressor.decompress_with_outliers(compressed)
    
    mse = np.mean((weights - restored)**2)
    mae = np.mean(np.abs(weights - restored))
    max_err = np.max(np.abs(weights - restored))
    
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"   Max Error: {max_err:.6f}")
    print(f"   Restored sample: {restored[:5]}")
    
    if mse < 0.05:
        print("\n✅ EXCELLENT! GPTQ-Lite works great!")
    elif mse < 0.2:
        print("\n✅ GOOD! Should work for LLMs.")
    else:
        print("\n⚠️  Needs more tuning.")
    
    print("="*70)

if __name__ == "__main__":
    test_gptq_lite()
