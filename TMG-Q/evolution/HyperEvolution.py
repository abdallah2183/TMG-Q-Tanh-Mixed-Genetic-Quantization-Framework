import numpy as np
import random
import copy
import time
import math

# --- إعدادات الحوسبة الفائقة ---
POPULATION_SIZE = 1000    # جيش من المعادلات
SAMPLE_SIZE = 10000       # حجم البيانات التي نختبر عليها (كبير جداً لدقة حقيقية)
MAX_DEPTH = 6             # عمق الشجرة (لا نريد معادلات أعقد من اللازم للهاتف)

# --- محاكاة أوزان LLM حقيقية (FP32) ---
# توزيع طبيعي + بعض الشواذ (Outliers) لمحاكاة الواقع الصعب
print("Generating 70B Model Implementation Data...")
REAL_WEIGHTS = np.concatenate([
    np.random.normal(0, 1.0, int(SAMPLE_SIZE * 0.9)),
    np.random.normal(5, 2.0, int(SAMPLE_SIZE * 0.1)) # Outliers
]).astype(np.float32)
np.random.shuffle(REAL_WEIGHTS)

# العمليات المتجهة (Vectorized Operations)
# هذه العمليات تعمل على 10000 رقم دفعة واحدة
OPS = ['+', '-', '*', '/', 'abs', 'sign', 'floor', 'min_w', 'max_w']

class HyperAlgo:
    def __init__(self):
        self.compress_tree = self.random_tree(depth=2)
        self.decompress_tree = self.random_tree(depth=2)
        self.fitness = -float('inf')
        self.code_str = ""

    def random_tree(self, depth=0):
        if depth >= MAX_DEPTH or (depth > 1 and random.random() < 0.3):
            return random.choice(['w', 'c']) # w: weights, c: constant
        
        op = random.choice(OPS)
        if op in ['abs', 'sign', 'floor', 'min_w', 'max_w']:
            return (op, self.random_tree(depth + 1))
        else:
            return (op, self.random_tree(depth + 1), self.random_tree(depth + 1))

    def eval_vectorized(self, tree, w_array, constants):
        """
        تنفيذ المعادلة على مصفوفة كاملة بسرعة البرق
        """
        if tree == 'w': return w_array
        if tree == 'c': return constants # ثابت عشوائي (Scale)
        
        op = tree[0]
        try:
            # Unary Ops
            if len(tree) == 2:
                val = self.eval_vectorized(tree[1], w_array, constants)
                if op == 'abs': return np.abs(val)
                if op == 'sign': return np.sign(val)
                if op == 'floor': return np.floor(val)
                if op == 'min_w': return np.min(val) # قيمة صغرى للمصفوفة كلها
                if op == 'max_w': return np.max(val) # قيمة عظمى
            
            # Binary Ops
            else:
                l = self.eval_vectorized(tree[1], w_array, constants)
                r = self.eval_vectorized(tree[2], w_array, constants)
                if op == '+': return l + r
                if op == '-': return l - r
                if op == '*': return l * r
                if op == '/': 
                    # حماية من القسمة على صفر
                    return np.divide(l, r, out=np.zeros_like(l), where=np.abs(r) > 1e-6)
        except:
            return np.zeros_like(w_array)
        
        return np.zeros_like(w_array)

    def to_string(self, tree):
        if tree == 'w': return "W"
        if tree == 'c': return "C"
        op = tree[0]
        if len(tree) == 2: return f"{op}({self.to_string(tree[1])})"
        return f"({self.to_string(tree[1])} {op} {self.to_string(tree[2])})"

def run_hyper_evolution():
    print(f"--- HYPER EVOLUTION ENGINE STARTED ---")
    print(f"Population: {POPULATION_SIZE} | Data Points: {SAMPLE_SIZE}")
    print("Press Ctrl+C to stop and take the best formula.\n")
    
    population = [HyperAlgo() for _ in range(POPULATION_SIZE)]
    
    # Stagnation Tracking
    best_ever_fitness = -float('inf')
    generations_since_improvement = 0
    PATIENCE_LIMIT = 200 
    
    generation = 1
    start_time = time.time()

    try:
        while True:
            # ... (Evaluation Logic) ...
            # [Code from previous step remains same until sorting]
            
            # --- TURBO MODE: Mini-Batch Sampling ---
            batch_idx = np.random.choice(len(REAL_WEIGHTS), 512) 
            batch_data = REAL_WEIGHTS[batch_idx]
            
            best_in_gen = None
            best_fitness = -float('inf')
            const = 0.5 
            
            for algo in population:
                raw_q = algo.eval_vectorized(algo.compress_tree, batch_data, const)
                q_int = np.clip(np.round(raw_q), -8, 7)
                restored = algo.eval_vectorized(algo.decompress_tree, q_int, const)
                mse = np.mean((batch_data - restored)**2)
                algo.fitness = -mse 
                
                if algo.fitness > best_fitness:
                    best_fitness = algo.fitness

            population.sort(key=lambda x: x.fitness, reverse=True)
            global_best = population[0]

            # --- EXTINCTION MONITOR ---
            if global_best.fitness > best_ever_fitness + 0.0001:
                best_ever_fitness = global_best.fitness
                generations_since_improvement = 0 # وجدنا حلاً أفضل، صفر العداد
            else:
                generations_since_improvement += 1
            
            if generations_since_improvement > PATIENCE_LIMIT:
                print(f"\n>>> EXTINCTION EVENT TRIGGERED! (Stuck for {PATIENCE_LIMIT} gens) <<<")
                print(">>> Killing 90% of population and introducing Aliens...")
                
                # Keep Top 5 Elites Only
                survivors = population[:5]
                # Fill the rest with fresh random DNA
                aliens = [HyperAlgo() for _ in range(POPULATION_SIZE - 5)]
                population = survivors + aliens
                
                generations_since_improvement = 0 # Reset
                generation += 1
                continue # Skip normal reproduction this turn

            # ... (Reporting Logic) ...
            
            # عرض التقدم كل 50 جيل (مع فحص كامل للدقة)
            if generation % 50 == 0:
                # فحص حقيقي على كل البيانات (Full Validation)
                full_cons = 0.5
                f_q = global_best.eval_vectorized(global_best.compress_tree, REAL_WEIGHTS, full_cons)
                f_int = np.clip(np.round(f_q), -8, 7)
                f_res = global_best.eval_vectorized(global_best.decompress_tree, f_int, full_cons)
                true_mse = np.mean((REAL_WEIGHTS - f_res)**2)
                
                elapsed = time.time() - start_time
                print(f"Gen {generation} | True Error: {true_mse:.6f} | Time: {elapsed:.1f}s")
                if generation % 100 == 0:
                    print(f"   Comp:   {global_best.to_string(global_best.compress_tree)}")
                    print(f"   DeComp: {global_best.to_string(global_best.decompress_tree)}")
                    print("-" * 40)
            
            # --- NEXT GENERATION (Tournament Selection & Mutation) ---
            next_gen = population[:50] # Top 50 Elite (Keep them safe)
            
            while len(next_gen) < POPULATION_SIZE:
                # Tournament Selection
                p1 = random.choice(population[:200])
                p2 = random.choice(population[:200])
                parent = p1 if p1.fitness > p2.fitness else p2
                
                child = copy.deepcopy(parent)
                
                # Aggressive Mutation (High Rate for discovery)
                if random.random() < 0.4:
                    # استبدال جزء كامل من الشجرة
                    if random.random() < 0.5:
                        child.compress_tree = child.random_tree(depth=random.randint(0, 4))
                    else:
                        child.decompress_tree = child.random_tree(depth=random.randint(0, 4))
                
                next_gen.append(child)
            
            population = next_gen
            generation += 1
            
    except KeyboardInterrupt:
        print("\n\n>>> DISCOVERY HALTED BY USER <<<")
        print("Best Formula Found:")
        print(f"Compress:   {population[0].to_string(population[0].compress_tree)}")
        print(f"Decompress: {population[0].to_string(population[0].decompress_tree)}")
        print(f"Final MSE Error: {-population[0].fitness:.6f}")

if __name__ == "__main__":
    run_hyper_evolution()
