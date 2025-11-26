import time
import numpy as np
from gpu_priors import GPUPriorGenerator
from sklearn.linear_model import LinearRegression

def cpu_version(num_datasets, rows):
    results_x = []
    results_y = []
    
    # We use a loop because each dataset has different random coefficients
    # This loop is what kills CPU performance in real-world complex priors
    for _ in range(num_datasets):
        X = np.random.rand(rows, 4).astype(np.float32)
        coeffs = np.random.randn(4).astype(np.float32)
        
        # Simple Logic for CPU benchmark
        y = (X[:, 0] * coeffs[0]) + (X[:, 1] * coeffs[1]) + \
            (X[:, 2] * coeffs[2]) - (X[:, 3] * coeffs[3])
                
        results_x.append(X)
        results_y.append(y)
    return results_x, results_y

def run_tests():
    # --- CONFIGURATION ---
    # SCALE UP! 100,000 datasets * 256 rows = 25,600,000 rows (~1 GB of data)
    # If your PC has < 16GB RAM, lower this to 50,000
    N_DATASETS = 100_000 
    ROWS = 256
    
    print(f"--- SETTING UP: {N_DATASETS} Datasets on RTX 3080 ---")
    print(f"Total Data Points: {N_DATASETS * ROWS:,} rows.")
    
    # Init Generator
    gen = GPUPriorGenerator(num_datasets=N_DATASETS, rows_per_dataset=ROWS)
    
    # --- TEST 1: GPU BENCHMARK ---
    print("\n--- GPU RUN ---")
    
    # Phase A: Computation (What actually matters for DL Training)
    start_compute = time.perf_counter()
    d_x, d_y = gen.generate()
    end_compute = time.perf_counter()
    
    # Phase B: Memory Transfer (Optional in real DL pipelines)
    start_transfer = time.perf_counter()
    h_x = d_x.copy_to_host() 
    h_y = d_y.copy_to_host()
    end_transfer = time.perf_counter()
    
    gpu_compute_time = end_compute - start_compute
    gpu_total_time = end_transfer - start_compute
    
    print(f"GPU Compute Time:  {gpu_compute_time:.4f} seconds (Pure Generation)")
    print(f"GPU Transfer Time: {end_transfer - start_transfer:.4f} seconds")
    print(f"GPU Total Time:    {gpu_total_time:.4f} seconds")

    # --- TEST 2: CPU BENCHMARK ---
    print("\n--- CPU RUN ---")
    # We run a smaller subset because Python loops are slow
    cpu_n = 5000 
    print(f"Benchmarking CPU on {cpu_n} datasets...")
    
    start_cpu = time.perf_counter()
    cpu_version(cpu_n, ROWS)
    end_cpu = time.perf_counter()
    
    cpu_subset_time = end_cpu - start_cpu
    cpu_projected = cpu_subset_time * (N_DATASETS / cpu_n)
    
    print(f"CPU Time (Subset):    {cpu_subset_time:.4f} seconds")
    print(f"CPU Time (Projected): {cpu_projected:.4f} seconds")

    # --- RESULTS ---
    print("\n" + "="*40)
    print(f"SPEEDUP (Pure Compute): {cpu_projected / gpu_compute_time:.1f}x FASTER")
    print(f"SPEEDUP (With IO):      {cpu_projected / gpu_total_time:.1f}x FASTER")
    print("="*40)
    
    # Interview Talking Point
    print("\n[Talking Point] In a Tabular Foundation Model pipeline, data stays on the GPU.")
    print(f"This means you can generate {N_DATASETS / gpu_compute_time:,.0f} datasets per second.")

if __name__ == "__main__":
    run_tests()