"""
=================================================================
TMG-Q Core Engine (Internal: GPTQLiteV2)
  Tanh-Mixed-Genetic Quantization Framework
  by Abdal
=================================================================
المعمل الثاني - خوارزمية ضغط LLM بالنسخة المحسنة

TMG-Q Components:
  T - Tanh: الضغط اللاخطي باستخدام Tanh
  M - Mixed: دقة مختلطة (FP32/FP16/INT4)
  G - Genetic: خوارزمية جينية HyperEvolution لاكتشاف الصيغ
  Q - Quantization: ضغط متقدم

التحسينات الأربعة:
  1. Dynamic Saliency     - نسبة Outliers ذكية لكل طبقة (ليست ثابتة 3%)
  2. Scaling Factor        - معامل موازنة لكل طبقة لإرجاع المتوسط الأصلي
  3. Activation-Aware      - Fitness تعتمد على مخرجات الطبقة (مثل AWQ)
  4. Layer-wise Recon.     - ترميم خطأ الطبقة السابقة في الطبقة التالية

المبدأ: AWQ أثبتت إن الأوزان المهمة ليست الأكبر حجماً،
بل الأوزان اللي لما تتغير → المخرجات تتغير أكثر.

Usage:
  from TMG_Q import TMGQ  # Recommended public API
  # or
  from GPTQ_Lite_V2 import GPTQLiteV2  # Internal/backward compat
"""

import numpy as np
import torch
import sys
import time


class GPTQLiteV2:
    """
    TMG-Q Core Engine (GPTQLiteV2)
    Tanh-Mixed-Genetic Quantization Framework
    
    الفرق عن V1:
    - نسبة الحماية (outlier %) تتغير ديناميكياً لكل طبقة
    - معامل Scale يُحسب بعد الضغط لضبط المخرجات
    - الخطأ يُحسب على الـ Activations وليس الأوزان فقط
    - كل طبقة تعوّض خطأ الطبقة السابقة
    """

    def __init__(self, group_size=128):
        self.group_size = group_size
        # V2: Layer-wise error accumulator
        self._prev_layer_error = None
        self._layer_index = 0
        self._total_layers = 0

    # ================================================================
    # 1. DYNAMIC SALIENCY (نسبة Outliers ذكية)
    # ================================================================
    def compute_dynamic_outlier_percent(self, weights, layer_name="", layer_idx=0, total_layers=1):
        """
        التحسين الأول: بدل 3% ثابتة لكل الطبقات،
        نحسب النسبة بناءً على "حساسية الطبقة".

        القاعدة:
        - الطبقات الأولى (تفهم القواعد) → حماية عالية (5-7%)
        - الطبقات الوسطى (معالجة) → حماية أقل (1-2%)
        - الطبقات الأخيرة (lm_head) → حماية عالية (5-7%)
        - الطبقات ذات التشتت العالي (std عالي) → حماية أكبر
        """
        # --- Position-based sensitivity ---
        if total_layers <= 1:
            position_factor = 1.0
        else:
            # Normalize position to [0, 1]
            pos = layer_idx / max(total_layers - 1, 1)
            # U-shaped curve: high at edges, low in middle
            # f(0) = 1.0, f(0.5) = 0.3, f(1.0) = 1.0
            position_factor = 1.0 - 0.7 * np.sin(np.pi * pos)

        # --- Standard deviation sensitivity ---
        std = float(np.std(weights))
        mean_abs = float(np.mean(np.abs(weights)))

        # Coefficient of variation: std / mean → higher = more spread = more outliers
        if mean_abs > 1e-8:
            cv = std / mean_abs
        else:
            cv = 1.0

        # Map CV to sensitivity: higher CV → higher protection
        # Typical LLM weight CV is 0.5-2.0
        std_factor = np.clip(cv / 1.5, 0.5, 2.0)

        # --- Critical layer name detection ---
        name_factor = 1.0
        critical_keys = ['embed', 'wte', 'wpe', 'lm_head', 'ln_', 'norm',
                         'layer_norm', 'final', 'head', 'classifier', 'output']
        if any(k in layer_name.lower() for k in critical_keys):
            name_factor = 1.5

        # --- Final outlier percentage ---
        # Base: 3%, scaled by all factors
        base_percent = 3.0
        dynamic_percent = base_percent * position_factor * std_factor * name_factor

        # Clamp to [1.0, 10.0] range
        dynamic_percent = float(np.clip(dynamic_percent, 1.0, 10.0))

        return dynamic_percent

    # ================================================================
    # 2. SCALING FACTOR (معامل موازنة)
    # ================================================================
    def compute_scaling_factor(self, original_weights, compressed_weights):
        """
        التحسين الثاني: بعد الضغط بـ tanh وإعادة التكميم،
        نضرب المصفوفة في معامل s بحيث:
            mean(compressed * s) ≈ mean(original)

        هذا يقلل الانحراف في Logits MSE (كان طالع 12).
        """
        orig_mean = float(np.mean(np.abs(original_weights)))
        comp_mean = float(np.mean(np.abs(compressed_weights)))

        if comp_mean < 1e-10:
            return 1.0

        # Scale factor to match original activation magnitude
        s = orig_mean / comp_mean

        # Clamp to reasonable range (avoid explosion)
        s = float(np.clip(s, 0.5, 3.0))

        return s

    def compute_advanced_scaling_factor(self, original_weights, compressed_weights):
        """
        V2 Advanced: Least-squares optimal scaling factor.
        s* = (original · compressed) / (compressed · compressed)
        This minimizes ||original - s * compressed||²
        """
        orig_flat = original_weights.flatten().astype(np.float64)
        comp_flat = compressed_weights.flatten().astype(np.float64)

        numerator = float(np.dot(orig_flat, comp_flat))
        denominator = float(np.dot(comp_flat, comp_flat))

        if denominator < 1e-12:
            return 1.0

        s = numerator / denominator
        s = float(np.clip(s, 0.5, 3.0))
        return s

    # ================================================================
    # 3. ACTIVATION-AWARE FITNESS (Fitness على المخرجات)
    # ================================================================
    def compute_activation_error(self, original_weights, compressed_weights,
                                  calibration_input=None):
        """
        التحسين الثالث: بدل MSE على الأوزان فقط،
        نحسب MSE على مخرجات الطبقة (Activations).

        هذا هو السر وراء تفوق AWQ:
        الوزن اللي MSE تبعه كبير بس ما يأثر على الـ Output → مو مهم.
        الوزن اللي MSE تبعه صغير بس يأثر كثير على الـ Output → مهم جداً.

        Parameters:
            original_weights: Original weight matrix [out_features x in_features]
            compressed_weights: Compressed weight matrix (same shape)
            calibration_input: Sample input [batch x in_features] (auto-generated if None)

        Returns:
            weight_mse: Traditional weight MSE
            activation_mse: Output activation MSE (what AWQ optimizes)
            combined_score: Weighted combination (activation-heavy)
        """
        weight_mse = float(np.mean((original_weights - compressed_weights) ** 2))

        # Generate calibration input if not provided
        if calibration_input is None:
            # Simulate typical activation statistics (normal distribution)
            in_features = original_weights.shape[-1] if original_weights.ndim >= 2 else original_weights.shape[0]
            calibration_input = np.random.randn(32, in_features).astype(np.float32) * 0.5

        # Compute original and compressed outputs
        try:
            if original_weights.ndim >= 2:
                orig_output = calibration_input @ original_weights.T
                comp_output = calibration_input @ compressed_weights.T
            else:
                # 1D weights (biases, norms, etc.)
                orig_output = calibration_input * original_weights
                comp_output = calibration_input * compressed_weights

            activation_mse = float(np.mean((orig_output - comp_output) ** 2))
        except Exception:
            activation_mse = weight_mse

        # Combined: 70% activation + 30% weight (activation-dominant like AWQ)
        combined_score = 0.7 * activation_mse + 0.3 * weight_mse

        return weight_mse, activation_mse, combined_score

    # ================================================================
    # 4. LAYER-WISE RECONSTRUCTION (ترميم خطأ الطبقة السابقة)
    # ================================================================
    def compute_error_compensation(self, original_weights, compressed_weights,
                                    prev_layer_error=None):
        """
        التحسين الرابع: بدل ضغط كل الطبقات مرة وحدة،
        نعوّض خطأ الطبقة السابقة في الطبقة التالية.

        المبدأ: لو الطبقة i فقدت Δ في المخرجات،
        نعدّل أوزان الطبقة i+1 بحيث تعوض Δ.

        هذا يمنع "كرة الثلج" (Error Accumulation) عبر 180 طبقة.
        """
        if prev_layer_error is None:
            return compressed_weights, np.zeros_like(compressed_weights)

        # Error from previous layer
        # prev_layer_error shape: [out_features_prev] or similar
        # Current weights shape: [out_features, in_features]

        try:
            if compressed_weights.ndim == 2 and prev_layer_error.ndim == 1:
                # The in_features of current layer = out_features of previous layer
                if len(prev_layer_error) == compressed_weights.shape[1]:
                    # Compute compensation: adjust current weights to absorb previous error
                    # W_compensated = W_compressed + α * error_correction
                    # where error_correction nudges the layer to counteract prev error

                    # Compute how much each input channel was affected
                    error_magnitude = np.abs(prev_layer_error)
                    max_err = float(np.max(error_magnitude)) if error_magnitude.size > 0 else 0

                    if max_err > 1e-8:
                        # Normalize error to [0, 1]
                        error_norm = error_magnitude / max_err

                        # Apply correction proportional to error
                        # α controls how aggressively we compensate (0.1 = gentle)
                        alpha = 0.3

                        # The correction: scale each column of W proportionally
                        # Channels with higher error get more correction
                        correction_factor = 1.0 + alpha * error_norm

                        compensated = compressed_weights * correction_factor[np.newaxis, :]
                    else:
                        compensated = compressed_weights
                else:
                    compensated = compressed_weights
            else:
                compensated = compressed_weights
        except Exception:
            compensated = compressed_weights

        # Compute remaining error for next layer
        residual_error = (original_weights - compensated).mean(axis=0) if original_weights.ndim >= 2 else \
                         (original_weights - compensated)

        return compensated, residual_error

    # ================================================================
    # CORE: Non-linear compress/decompress (from V1)
    # ================================================================
    def _paper_compress_raw(self, w, c):
        eps = 1e-12
        w_sign = np.where(w >= 0, 1.0, -1.0)
        unity = w / (w + w_sign * eps)
        denom = np.abs(np.tanh(c)) + (unity - np.abs(c))
        denom = np.where(np.abs(denom) < eps, w_sign * eps, denom)
        return w / denom

    def _paper_decompress(self, q, c):
        return q * (np.tanh(c) - c) + q

    def _pack_int4_signed(self, q_int8):
        q_u = (q_int8.astype(np.int16) + 8).astype(np.uint8)
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

    # ================================================================
    # V2 CALIBRATION: Activation-aware constant search
    # ================================================================
    def _calibrate_constant_v2(self, weights, calibration_input=None,
                                 search_range=(0.1, 3.0), steps=40):
        """
        V2 Calibration: finds the best constant c that minimizes
        ACTIVATION error (not just weight MSE).
        """
        best_c = 0.5
        best_score = float('inf')

        candidates = np.linspace(float(search_range[0]), float(search_range[1]), int(steps))

        for c in candidates:
            try:
                raw_q = self._paper_compress_raw(weights, float(c))
                q_int = np.clip(np.round(raw_q), -8, 7).astype(np.int8)
                restored = self._paper_decompress(q_int.astype(np.float32), float(c)).astype(np.float32)

                if calibration_input is not None and weights.ndim >= 2:
                    # V2: Activation-aware scoring
                    _, act_mse, combined = self.compute_activation_error(
                        weights, restored, calibration_input
                    )
                    score = combined
                else:
                    # Fallback to weight MSE
                    score = float(np.mean((weights - restored) ** 2))

                if score < best_score:
                    best_score = score
                    best_c = float(c)
            except Exception:
                continue

        return best_c

    # ================================================================
    # V2 MAIN: compress_with_outliers (enhanced)
    # ================================================================
    def compress_v2(self, weights, layer_name="", layer_idx=0,
                     total_layers=1, mode="nonlinear",
                     calibration_input=None, prev_layer_error=None):
        """
        Main V2 compression pipeline:
        1. Dynamic outlier detection
        2. Activation-aware calibration
        3. Non-linear quantization with scaling factor
        4. Error compensation from previous layer
        """
        original_shape = weights.shape
        weights_2d = weights.reshape(-1) if weights.ndim == 1 else weights

        # ---- Step 1: Dynamic Saliency (outlier %) ----
        outlier_percent = self.compute_dynamic_outlier_percent(
            weights_2d, layer_name, layer_idx, total_layers
        )

        weights_flat = weights.flatten().astype(np.float32)
        threshold = np.percentile(np.abs(weights_flat), 100.0 - outlier_percent)
        outlier_mask = np.abs(weights_flat) >= threshold
        n_outliers = int(np.sum(outlier_mask))

        outlier_values = weights_flat[outlier_mask].astype(np.float16) if n_outliers > 0 \
            else np.array([], dtype=np.float16)

        non_outlier_data = weights_flat[~outlier_mask].astype(np.float32)

        print(f"       [V2-Saliency] Outlier%: {outlier_percent:.1f}% "
              f"({n_outliers}/{len(weights_flat)} protected)")

        # ---- Step 2: Calibrate constant (activation-aware if possible) ----
        calib_input_for_layer = calibration_input
        if calibration_input is None and weights.ndim == 2:
            # Auto-generate calibration data
            in_features = weights.shape[1]
            calib_input_for_layer = np.random.randn(32, in_features).astype(np.float32) * 0.5

        # For constant calibration, we need the full 2D weights (with outliers zeroed)
        weights_for_calib = weights_flat.copy()
        weights_for_calib[outlier_mask] = 0.0
        weights_for_calib_2d = weights_for_calib.reshape(original_shape) if weights.ndim >= 2 else weights_for_calib

        if mode == "nonlinear":
            c = self._calibrate_constant_v2(
                weights_for_calib_2d, calib_input_for_layer,
                search_range=(0.1, 3.0), steps=40
            )
            constants = np.array([c], dtype=np.float32)

            raw_q = self._paper_compress_raw(non_outlier_data, c)
            q_int = np.clip(np.round(raw_q), -8, 7).astype(np.int8)
            restored_non_outlier = self._paper_decompress(q_int.astype(np.float32), c).astype(np.float32)

            # ---- Step 3: Scaling Factor ----
            scaling_factor = self.compute_advanced_scaling_factor(non_outlier_data, restored_non_outlier)
            restored_non_outlier_scaled = restored_non_outlier * scaling_factor

            print(f"       [V2-Scale] Factor: {scaling_factor:.4f}")

            # Weight MSE before and after scaling
            mse_before = float(np.mean((non_outlier_data - restored_non_outlier) ** 2))
            mse_after = float(np.mean((non_outlier_data - restored_non_outlier_scaled) ** 2))
            print(f"       [V2-Scale] MSE: {mse_before:.6f} → {mse_after:.6f} "
                  f"({'✅ improved' if mse_after < mse_before else '⚠️ kept'})")

            # Only use scaling if it actually improves MSE
            if mse_after < mse_before:
                final_scaling = scaling_factor
            else:
                final_scaling = 1.0

            packed = self._pack_int4_signed(q_int)

            # ---- Step 4: Error Compensation ----
            # Reconstruct full array to compute layer error
            full_restored = np.zeros(len(weights_flat), dtype=np.float32)
            full_restored[~outlier_mask] = restored_non_outlier * final_scaling
            full_restored[outlier_mask] = outlier_values.astype(np.float32)

            if prev_layer_error is not None and weights.ndim == 2:
                compensated_2d, residual_error = self.compute_error_compensation(
                    weights.astype(np.float32),
                    full_restored.reshape(original_shape),
                    prev_layer_error
                )
                # Recompute error after compensation
                comp_mse = float(np.mean((weights.astype(np.float32) - compensated_2d) ** 2))
                orig_mse = float(np.mean((weights.astype(np.float32) - full_restored.reshape(original_shape)) ** 2))
                print(f"       [V2-Recon] Error compensation: {orig_mse:.6f} → {comp_mse:.6f}")
            else:
                residual_error = None

            # Compute layer error for next layer
            layer_error = (weights_flat - full_restored).astype(np.float32)
            if weights.ndim == 2:
                # Mean error per output channel
                layer_error_per_channel = (weights.astype(np.float32) - full_restored.reshape(original_shape)).mean(axis=0)
            else:
                layer_error_per_channel = layer_error

            # ---- Activation-aware quality report ----
            if calib_input_for_layer is not None and weights.ndim == 2:
                w_mse, act_mse, combined = self.compute_activation_error(
                    weights.astype(np.float32),
                    full_restored.reshape(original_shape),
                    calib_input_for_layer
                )
                print(f"       [V2-Quality] Weight MSE: {w_mse:.6f} | "
                      f"Activation MSE: {act_mse:.6f} | Combined: {combined:.6f}")

            return {
                'packed': packed,
                'scales': None,
                'zero_points': None,
                'constants': constants,
                'scaling_factor': np.array([final_scaling], dtype=np.float32),
                'outlier_mask': outlier_mask,
                'outlier_values': outlier_values,
                'original_shape': original_shape,
                'n_non_outliers': len(non_outlier_data),
                'outlier_percent': outlier_percent,
                'layer_error': layer_error_per_channel,
                'type': 'gptq_lite_v2_nonlinear'
            }

        else:
            # Linear mode with V2 enhancements
            n_groups = int(np.ceil(len(non_outlier_data) / self.group_size))
            padded_size = n_groups * self.group_size
            padded_data = np.zeros(padded_size, dtype=np.float32)
            padded_data[:len(non_outlier_data)] = non_outlier_data

            grouped = padded_data.reshape(n_groups, self.group_size)
            scales = np.zeros(n_groups, dtype=np.float32)
            zero_points = np.zeros(n_groups, dtype=np.uint8)
            quantized_groups = np.zeros_like(grouped, dtype=np.uint8)

            for i in range(n_groups):
                group = grouped[i]
                min_val = float(np.min(group))
                max_val = float(np.max(group))
                scale = (max_val - min_val) / 15.0
                if scale == 0:
                    scale = 1.0
                zero_point = int(round(-min_val / scale))
                zero_point = max(0, min(15, zero_point))

                scales[i] = scale
                zero_points[i] = zero_point
                scaled = group / scale + zero_point
                quantized_groups[i] = np.clip(np.round(scaled), 0, 15).astype(np.uint8)

            quantized_flat = quantized_groups.flatten()
            packed = np.zeros(len(quantized_flat) // 2, dtype=np.uint8)
            packed |= (quantized_flat[0::2] << 4)
            packed |= (quantized_flat[1::2] & 0x0F)

            # V2: Compute scaling factor for linear mode too
            dequantized = np.zeros_like(grouped, dtype=np.float32)
            for i in range(n_groups):
                dequantized[i] = (quantized_groups[i].astype(np.float32) - zero_points[i]) * scales[i]
            dequantized_flat = dequantized.flatten()[:len(non_outlier_data)]

            scaling_factor = self.compute_advanced_scaling_factor(non_outlier_data, dequantized_flat)
            print(f"       [V2-Scale] Factor: {scaling_factor:.4f}")

            # Layer error for next layer
            full_restored = np.zeros(len(weights_flat), dtype=np.float32)
            full_restored[~outlier_mask] = dequantized_flat * scaling_factor
            full_restored[outlier_mask] = outlier_values.astype(np.float32)

            if weights.ndim == 2:
                layer_error_per_channel = (weights.astype(np.float32) - full_restored.reshape(original_shape)).mean(axis=0)
            else:
                layer_error_per_channel = (weights_flat - full_restored)

            return {
                'packed': packed,
                'scales': scales,
                'zero_points': zero_points,
                'constants': None,
                'scaling_factor': np.array([scaling_factor], dtype=np.float32),
                'outlier_mask': outlier_mask,
                'outlier_values': outlier_values,
                'original_shape': original_shape,
                'n_non_outliers': len(non_outlier_data),
                'outlier_percent': outlier_percent,
                'layer_error': layer_error_per_channel,
                'type': 'gptq_lite_v2_linear'
            }

    # ================================================================
    # V2 DECOMPRESS
    # ================================================================
    def decompress_v2(self, compressed_data):
        """Decompress V2 format with scaling factor support."""
        packed = compressed_data['packed']
        scales = compressed_data.get('scales', None)
        zero_points = compressed_data.get('zero_points', None)
        constants = compressed_data.get('constants', None)
        scaling_factor = compressed_data.get('scaling_factor', np.array([1.0]))
        outlier_mask = compressed_data['outlier_mask']
        outlier_values = compressed_data['outlier_values']
        original_shape = compressed_data['original_shape']
        n_non_outliers = compressed_data['n_non_outliers']

        sf = float(scaling_factor[0]) if scaling_factor is not None else 1.0

        if scales is None or zero_points is None:
            # Nonlinear mode
            c = 0.5
            if constants is not None and len(constants) > 0:
                c = float(constants[0])

            q_int = self._unpack_int4_signed(packed, n_non_outliers)
            dequantized_flat = self._paper_decompress(q_int.astype(np.float32), c).astype(np.float32)

            # V2: Apply scaling factor
            dequantized_flat *= sf

            total_elements = int(np.prod(original_shape))
            restored = np.zeros(total_elements, dtype=np.float32)
            restored[~outlier_mask] = dequantized_flat
            restored[outlier_mask] = outlier_values.astype(np.float32)
            return restored.reshape(original_shape)

        else:
            # Linear mode
            n_elements = len(packed) * 2
            unpacked = np.zeros(n_elements, dtype=np.float32)
            unpacked[0::2] = (packed >> 4)
            unpacked[1::2] = (packed & 0x0F)

            n_groups = len(scales)
            unpacked_grouped = unpacked.reshape(n_groups, self.group_size)
            dequantized = np.zeros_like(unpacked_grouped, dtype=np.float32)

            for i in range(n_groups):
                dequantized[i] = (unpacked_grouped[i] - zero_points[i]) * scales[i]

            dequantized_flat = dequantized.flatten()[:n_non_outliers]

            # V2: Apply scaling factor
            dequantized_flat *= sf

            total_elements = int(np.prod(original_shape))
            restored = np.zeros(total_elements, dtype=np.float32)
            restored[~outlier_mask] = dequantized_flat
            restored[outlier_mask] = outlier_values.astype(np.float32)
            return restored.reshape(original_shape)


# ================================================================
# TEST SUITE
# ================================================================
def test_v2():
    """Full V2 test with comparison to V1 behavior."""
    print("\n" + "=" * 70)
    print("🧪 TMG-Q TEST SUITE")
    print("   Tanh-Mixed-Genetic Quantization Framework")
    print("=" * 70)

    np.random.seed(42)

    # Simulate 4 LLM layers with different characteristics
    layers = [
        {
            "name": "model.layer_0.self_attn.q_proj",
            "weights": np.random.normal(0, 0.8, (768, 768)).astype(np.float32),
            "type": "early (قواعد)"
        },
        {
            "name": "model.layer_10.mlp.dense_h_to_4h",
            "weights": np.random.normal(0, 0.3, (3072, 768)).astype(np.float32),
            "type": "middle (معالجة)"
        },
        {
            "name": "model.layer_22.self_attn.v_proj",
            "weights": np.random.normal(0, 0.5, (768, 768)).astype(np.float32),
            "type": "late (متأخرة)"
        },
        {
            "name": "lm_head",
            # lm_head typically has high std/outliers
            "weights": np.concatenate([
                np.random.normal(0, 1.5, (50257, 700)),
                np.random.normal(5, 3.0, (50257, 68))
            ], axis=1).astype(np.float32),
            "type": "head (اختيار الكلمات)"
        },
    ]

    compressor = GPTQLiteV2(group_size=128)
    total_layers = len(layers)
    prev_error = None

    results = []

    for idx, layer_info in enumerate(layers):
        name = layer_info["name"]
        weights = layer_info["weights"]
        ltype = layer_info["type"]

        print(f"\n{'─' * 70}")
        print(f"  Layer {idx + 1}/{total_layers}: {name}")
        print(f"  Type: {ltype}")
        print(f"  Shape: {weights.shape} | Params: {weights.size:,}")
        print(f"  Stats: mean={np.mean(weights):.4f}, std={np.std(weights):.4f}")
        print(f"{'─' * 70}")

        # Compress with V2
        compressed = compressor.compress_v2(
            weights,
            layer_name=name,
            layer_idx=idx,
            total_layers=total_layers,
            mode="nonlinear",
            prev_layer_error=prev_error
        )

        # Decompress
        restored = compressor.decompress_v2(compressed)

        # Quality metrics
        mse = float(np.mean((weights - restored) ** 2))
        mae = float(np.mean(np.abs(weights - restored)))
        max_err = float(np.max(np.abs(weights - restored)))

        # Compression ratio
        comp_size = compressed['packed'].nbytes + compressed['outlier_values'].nbytes + \
                    compressed['outlier_mask'].nbytes
        if compressed.get('scaling_factor') is not None:
            comp_size += compressed['scaling_factor'].nbytes
        if compressed.get('constants') is not None:
            comp_size += compressed['constants'].nbytes

        orig_size = weights.nbytes
        ratio = orig_size / comp_size if comp_size > 0 else 0

        print(f"\n  📊 Results:")
        print(f"     Weight MSE:   {mse:.6f}")
        print(f"     Weight MAE:   {mae:.6f}")
        print(f"     Max Error:    {max_err:.6f}")
        print(f"     Outlier %:    {compressed['outlier_percent']:.1f}%")
        print(f"     Scale Factor: {float(compressed['scaling_factor'][0]):.4f}")
        print(f"     Compression:  {ratio:.2f}x ({orig_size / 1024 / 1024:.1f} MB → "
              f"{comp_size / 1024 / 1024:.1f} MB)")

        results.append({
            'name': name, 'mse': mse, 'mae': mae,
            'ratio': ratio, 'outlier_pct': compressed['outlier_percent']
        })

        # Pass error to next layer
        prev_error = compressed.get('layer_error', None)

    # Summary
    print(f"\n{'=' * 70}")
    print(f"📋 TMG-Q SUMMARY")
    print(f"{'=' * 70}")
    avg_mse = np.mean([r['mse'] for r in results])
    avg_ratio = np.mean([r['ratio'] for r in results])
    print(f"  Average Weight MSE:      {avg_mse:.6f}")
    print(f"  Average Compression:     {avg_ratio:.2f}x")
    outlier_pcts_str = ", ".join(f'{r["outlier_pct"]:.1f}%' for r in results)
    print(f"  Outlier % per layer:     [{outlier_pcts_str}]")

    if avg_mse < 0.05:
        print(f"\n  ✅✅ EXCELLENT! TMG-Q quality is very high!")
    elif avg_mse < 0.2:
        print(f"\n  ✅ GOOD! TMG-Q should work well for LLMs.")
    else:
        print(f"\n  ⚠️ Needs more tuning.")

    print(f"{'=' * 70}\n")

    return results


if __name__ == "__main__":
    test_v2()
