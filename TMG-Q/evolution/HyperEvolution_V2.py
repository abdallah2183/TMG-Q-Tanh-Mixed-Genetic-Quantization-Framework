"""
=================================================================
HyperEvolution V2: Activation-Aware Genetic Algorithm
=================================================================
المحرك الجيني المحسن - يبحث عن معادلات ضغط/فك
بناءً على خطأ المخرجات (Activations) وليس الأوزان فقط.

التحسين الرئيسي (من خطة Gemini):
  - V1: fitness = -MSE(weights, restored_weights)
  - V2: fitness = -MSE(W @ X, W_compressed @ X)  ← هذا سر AWQ!
  + معامل Scale يتطور جينياً مع المعادلة
  + Calibration data حقيقي
"""

import numpy as np
import random
import copy
import time
import math

# --- إعدادات ---
POPULATION_SIZE = 800
SAMPLE_SIZE = 10000
MAX_DEPTH = 6
ELITE_SIZE = 50
TOURNAMENT_SIZE = 200
MUTATION_RATE = 0.4
PATIENCE_LIMIT = 250

# --- محاكاة أوزان LLM حقيقية ---
print("🧬 HyperEvolution V2: Generating realistic LLM weight data...")
np.random.seed(42)
REAL_WEIGHTS_FLAT = np.concatenate([
    np.random.normal(0, 0.5, int(SAMPLE_SIZE * 0.85)),   # Normal weights
    np.random.normal(0, 1.5, int(SAMPLE_SIZE * 0.10)),   # High-std weights
    np.random.normal(5, 2.0, int(SAMPLE_SIZE * 0.05)),   # Outliers
]).astype(np.float32)
np.random.shuffle(REAL_WEIGHTS_FLAT)

# Reshape to 2D matrix for activation-aware scoring
MATRIX_ROWS = 100  # ~output features
MATRIX_COLS = SAMPLE_SIZE // MATRIX_ROWS
REAL_WEIGHTS_2D = REAL_WEIGHTS_FLAT[:MATRIX_ROWS * MATRIX_COLS].reshape(MATRIX_ROWS, MATRIX_COLS)

# Generate calibration input (simulates real activations flowing through the network)
CALIBRATION_INPUT = np.random.randn(16, MATRIX_COLS).astype(np.float32) * 0.5

# Original output (ground truth for fitness)
ORIGINAL_OUTPUT = CALIBRATION_INPUT @ REAL_WEIGHTS_2D.T  # [16, 100]

print(f"   Weights: {REAL_WEIGHTS_2D.shape} ({REAL_WEIGHTS_2D.size:,} params)")
print(f"   Calibration: {CALIBRATION_INPUT.shape}")
print(f"   Target Output: {ORIGINAL_OUTPUT.shape}")

# Operations
OPS = ['+', '-', '*', '/', 'abs', 'sign', 'floor', 'clip', 'tanh_like']


class HyperAlgoV2:
    """
    Genetic algorithm organism that evolves compression formulas.
    V2: Includes a learnable scale factor per organism.
    """
    def __init__(self):
        self.compress_tree = self.random_tree(depth=2)
        self.decompress_tree = self.random_tree(depth=2)
        self.scale_factor = np.random.uniform(0.8, 1.2)  # V2: Evolving scale
        self.fitness = -float('inf')
        self.weight_mse = float('inf')
        self.activation_mse = float('inf')

    def random_tree(self, depth=0):
        if depth >= MAX_DEPTH or (depth > 1 and random.random() < 0.3):
            return random.choice(['w', 'c'])
        op = random.choice(OPS)
        if op in ['abs', 'sign', 'floor', 'clip', 'tanh_like']:
            return (op, self.random_tree(depth + 1))
        else:
            return (op, self.random_tree(depth + 1), self.random_tree(depth + 1))

    def eval_vectorized(self, tree, w_array, constants):
        """Execute formula tree on entire array (vectorized)."""
        if tree == 'w': return w_array
        if tree == 'c': return constants

        op = tree[0]
        try:
            if len(tree) == 2:
                val = self.eval_vectorized(tree[1], w_array, constants)
                if op == 'abs': return np.abs(val)
                if op == 'sign': return np.sign(val)
                if op == 'floor': return np.floor(val)
                if op == 'clip': return np.clip(val, -8, 7)
                if op == 'tanh_like': return np.tanh(val)  # Bounded non-linearity
            else:
                l = self.eval_vectorized(tree[1], w_array, constants)
                r = self.eval_vectorized(tree[2], w_array, constants)
                if op == '+': return l + r
                if op == '-': return l - r
                if op == '*': return l * r
                if op == '/':
                    return np.divide(l, r, out=np.zeros_like(l, dtype=np.float32),
                                     where=np.abs(r) > 1e-6)
        except:
            return np.zeros_like(w_array, dtype=np.float32)

        return np.zeros_like(w_array, dtype=np.float32)

    def to_string(self, tree):
        if tree == 'w': return "W"
        if tree == 'c': return "C"
        op = tree[0]
        if len(tree) == 2: return f"{op}({self.to_string(tree[1])})"
        return f"({self.to_string(tree[1])} {op} {self.to_string(tree[2])})"


def evaluate_organism_activation_aware(algo, weights_2d, calib_input, original_output, const=0.5):
    """
    V2 FITNESS FUNCTION:
    يحسب الخطأ بناءً على المخرجات (Activations) وليس الأوزان فقط.

    score = -(0.7 * activation_MSE + 0.3 * weight_MSE)

    هذا التغيير هو اللي خلى AWQ تتفوق عالمياً.
    """
    try:
        weights_flat = weights_2d.flatten().astype(np.float32)

        # Step 1: Compress
        raw_q = algo.eval_vectorized(algo.compress_tree, weights_flat, const)

        # Step 2: Quantize to INT4
        q_int = np.clip(np.round(raw_q), -8, 7).astype(np.float32)

        # Step 3: Decompress
        restored_flat = algo.eval_vectorized(algo.decompress_tree, q_int, const)

        # Step 4: Apply evolved scale factor
        restored_flat *= algo.scale_factor

        # Check for NaN/Inf
        if not np.isfinite(restored_flat).all():
            return -1e10, 1e10, 1e10

        # --- WEIGHT MSE ---
        weight_mse = float(np.mean((weights_flat - restored_flat) ** 2))

        # --- ACTIVATION MSE (the V2 magic) ---
        restored_2d = restored_flat.reshape(weights_2d.shape)
        compressed_output = calib_input @ restored_2d.T  # [batch, out_features]
        activation_mse = float(np.mean((original_output - compressed_output) ** 2))

        # --- COMBINED FITNESS (activation-dominant) ---
        combined = 0.7 * activation_mse + 0.3 * weight_mse
        fitness = -combined

        return fitness, weight_mse, activation_mse

    except Exception:
        return -1e10, 1e10, 1e10


def run_hyper_evolution_v2():
    """Main evolution loop with activation-aware fitness."""
    print(f"\n{'🧬' * 30}")
    print(f"  HYPER EVOLUTION V2 - ACTIVATION-AWARE")
    print(f"  Population: {POPULATION_SIZE} | Params: {REAL_WEIGHTS_2D.size:,}")
    print(f"  Fitness: 70% Activation MSE + 30% Weight MSE")
    print(f"{'🧬' * 30}")
    print("  Press Ctrl+C to stop and extract the best formula.\n")

    population = [HyperAlgoV2() for _ in range(POPULATION_SIZE)]

    best_ever_fitness = -float('inf')
    best_ever_algo = None
    generations_since_improvement = 0

    generation = 1
    start_time = time.time()

    # Use a mini-batch of the 2D weights for speed
    # Full evaluation every 50 generations
    batch_rows = min(50, MATRIX_ROWS)
    batch_cols = min(50, MATRIX_COLS)

    try:
        while True:
            # --- MINI-BATCH SAMPLING (for speed) ---
            row_idx = np.random.choice(MATRIX_ROWS, batch_rows, replace=False)
            col_idx = np.random.choice(MATRIX_COLS, batch_cols, replace=False)
            batch_weights = REAL_WEIGHTS_2D[np.ix_(row_idx, col_idx)]
            batch_calib = CALIBRATION_INPUT[:, col_idx]  # [16, batch_cols]
            batch_output = batch_calib @ batch_weights.T  # [16, batch_rows]

            const = 0.5

            # --- EVALUATE ALL ORGANISMS ---
            for algo in population:
                fitness, w_mse, a_mse = evaluate_organism_activation_aware(
                    algo, batch_weights, batch_calib, batch_output, const
                )
                algo.fitness = fitness
                algo.weight_mse = w_mse
                algo.activation_mse = a_mse

            # Sort by fitness (descending)
            population.sort(key=lambda x: x.fitness, reverse=True)
            champion = population[0]

            # --- STAGNATION DETECTION ---
            if champion.fitness > best_ever_fitness + 0.0001:
                best_ever_fitness = champion.fitness
                best_ever_algo = copy.deepcopy(champion)
                generations_since_improvement = 0
            else:
                generations_since_improvement += 1

            # --- EXTINCTION EVENT ---
            if generations_since_improvement > PATIENCE_LIMIT:
                print(f"\n  💀 EXTINCTION ({PATIENCE_LIMIT} gens stagnant)! Introducing aliens...")
                survivors = population[:5]
                aliens = [HyperAlgoV2() for _ in range(POPULATION_SIZE - 5)]
                population = survivors + aliens
                generations_since_improvement = 0
                generation += 1
                continue

            # --- REPORTING ---
            if generation % 25 == 0:
                elapsed = time.time() - start_time

                # Full validation every 50 gens
                if generation % 50 == 0 and best_ever_algo is not None:
                    f_q = best_ever_algo.eval_vectorized(
                        best_ever_algo.compress_tree, REAL_WEIGHTS_2D.flatten(), 0.5)
                    f_int = np.clip(np.round(f_q), -8, 7).astype(np.float32)
                    f_res = best_ever_algo.eval_vectorized(
                        best_ever_algo.decompress_tree, f_int, 0.5)
                    f_res *= best_ever_algo.scale_factor

                    true_w_mse = float(np.mean((REAL_WEIGHTS_2D.flatten() - f_res) ** 2))

                    f_res_2d = f_res.reshape(REAL_WEIGHTS_2D.shape)
                    true_act_output = CALIBRATION_INPUT @ f_res_2d.T
                    true_act_mse = float(np.mean((ORIGINAL_OUTPUT - true_act_output) ** 2))

                    print(f"  Gen {generation:5d} | "
                          f"W-MSE: {true_w_mse:.6f} | "
                          f"ACT-MSE: {true_act_mse:.6f} | "
                          f"Scale: {best_ever_algo.scale_factor:.4f} | "
                          f"Time: {elapsed:.0f}s")

                    if generation % 100 == 0:
                        print(f"    Compress:   {best_ever_algo.to_string(best_ever_algo.compress_tree)}")
                        print(f"    Decompress: {best_ever_algo.to_string(best_ever_algo.decompress_tree)}")
                        print(f"    {'─' * 50}")
                else:
                    print(f"  Gen {generation:5d} | "
                          f"Best Fitness: {champion.fitness:.6f} | "
                          f"W-MSE: {champion.weight_mse:.6f} | "
                          f"ACT-MSE: {champion.activation_mse:.6f} | "
                          f"Time: {elapsed:.0f}s")

            # --- REPRODUCTION ---
            next_gen = population[:ELITE_SIZE]

            while len(next_gen) < POPULATION_SIZE:
                # Tournament selection
                p1 = random.choice(population[:TOURNAMENT_SIZE])
                p2 = random.choice(population[:TOURNAMENT_SIZE])
                parent = p1 if p1.fitness > p2.fitness else p2

                child = copy.deepcopy(parent)

                # Mutate formula trees
                if random.random() < MUTATION_RATE:
                    if random.random() < 0.5:
                        child.compress_tree = child.random_tree(depth=random.randint(0, 4))
                    else:
                        child.decompress_tree = child.random_tree(depth=random.randint(0, 4))

                # V2: Mutate scale factor too
                if random.random() < 0.3:
                    child.scale_factor *= np.random.uniform(0.9, 1.1)
                    child.scale_factor = float(np.clip(child.scale_factor, 0.3, 3.0))

                next_gen.append(child)

            population = next_gen
            generation += 1

    except KeyboardInterrupt:
        print(f"\n\n{'=' * 70}")
        print(f"  🏆 EVOLUTION HALTED BY USER at Generation {generation}")
        print(f"{'=' * 70}")

        if best_ever_algo is not None:
            # Final full validation
            f_q = best_ever_algo.eval_vectorized(
                best_ever_algo.compress_tree, REAL_WEIGHTS_2D.flatten(), 0.5)
            f_int = np.clip(np.round(f_q), -8, 7).astype(np.float32)
            f_res = best_ever_algo.eval_vectorized(
                best_ever_algo.decompress_tree, f_int, 0.5)
            f_res *= best_ever_algo.scale_factor

            true_w_mse = float(np.mean((REAL_WEIGHTS_2D.flatten() - f_res) ** 2))
            f_res_2d = f_res.reshape(REAL_WEIGHTS_2D.shape)
            true_act_output = CALIBRATION_INPUT @ f_res_2d.T
            true_act_mse = float(np.mean((ORIGINAL_OUTPUT - true_act_output) ** 2))

            print(f"\n  🏆 BEST FORMULA DISCOVERED:")
            print(f"     Compress:   {best_ever_algo.to_string(best_ever_algo.compress_tree)}")
            print(f"     Decompress: {best_ever_algo.to_string(best_ever_algo.decompress_tree)}")
            print(f"     Scale:      {best_ever_algo.scale_factor:.6f}")
            print(f"\n  📊 FINAL METRICS:")
            print(f"     Weight MSE:     {true_w_mse:.6f}")
            print(f"     Activation MSE: {true_act_mse:.6f}")
            print(f"     Combined Score: {0.7 * true_act_mse + 0.3 * true_w_mse:.6f}")
            print(f"\n  💡 To use this formula in GPTQ_Lite_V2:")
            print(f"     Replace _paper_compress_raw() and _paper_decompress()")
            print(f"     with the discovered formulas above.")
        else:
            print("  ⚠️ No valid formula found yet.")

        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    run_hyper_evolution_v2()
