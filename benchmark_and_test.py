import time
import numpy as np
from gpu_priors import GPUPriorGenerator

def cpu_version_heavy(num_datasets, rows):
    results_x = []
    results_y = []
    
    for _ in range(num_datasets):
        X = np.random.uniform(-1, 1, (rows, 4)).astype(np.float32)
        coeffs = np.random.randn(4).astype(np.float32)
        
        # THE BOTTLENECK:
        # Numpy has to calculate Sin/Cos/Tanh/Exp for every single number.
        term1 = np.sin(X[:, 0] * coeffs[0] * 3.14)
        term2 = np.cos(X[:, 1] * coeffs[1] * 3.14)
        term3 = np.tanh(X[:, 2] * coeffs[2])
        term4 = np.exp(X[:, 3] * coeffs[3] * -1.0)
        
        y = term1 + term2 + term3 + term4
                
        results_x.append(X)
        results_y.append(y)
    return results_x, results_y

def run_tests():
    # SETTINGS: 100,000 Datasets
    N_DATASETS = 100_000 
    ROWS = 256
    
    print(f"--- SETTING UP: {N_DATASETS} Datasets on RTX 3080 ---")
    print("Mode: HIGH COMPLEXITY (Sin/Cos/Tanh/Exp)")
    gen = GPUPriorGenerator(num_datasets=N_DATASETS, rows_per_dataset=ROWS)
    
    # --- TEST 1: GPU ---
    print("\n--- GPU RUN ---")
    start_compute = time.perf_counter()
    d_x, d_y = gen.generate()
    end_compute = time.perf_counter()
    
    gpu_compute_time = end_compute - start_compute
    print(f"GPU Compute Time: {gpu_compute_time:.4f} s")

    # --- TEST 2: CPU ---
    print("\n--- CPU RUN ---")
    cpu_n = 2000 # Keep subset small, CPU will be slower now
    print(f"Benchmarking CPU on {cpu_n} datasets...")
    
    start_cpu = time.perf_counter()
    cpu_version_heavy(cpu_n, ROWS)
    end_cpu = time.perf_counter()
    
    cpu_subset_time = end_cpu - start_cpu
    cpu_projected = cpu_subset_time * (N_DATASETS / cpu_n)
    
    print(f"CPU Time (Projected): {cpu_projected:.4f} s")

    # --- RESULTS ---
    speedup = cpu_projected / gpu_compute_time
    print("\n" + "="*40)
    print(f"SPEEDUP: {speedup:.1f}x FASTER")
    print("="*40)

if __name__ == "__main__":
    run_tests()