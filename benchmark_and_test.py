import time
import numpy as np
from gpu_priors import GPUPriorGenerator
from sklearn.linear_model import LinearRegression

def cpu_version(num_datasets, rows):
    """
    Simulates the same logic as the GPU kernel using Numpy.
    """
    results_x = []
    results_y = []
    
    for _ in range(num_datasets):
        # 1. Generate Random Features (Same as GPU)
        X = np.random.rand(rows, 4).astype(np.float32)
        coeffs = np.random.randn(4).astype(np.float32)
        complexity_check = np.random.rand()
        
        # 2. Apply the Logic (Iterating this 10,000 times is the bottleneck)
        if complexity_check > 0.5:
            # Complex Path: sin(x1) + x2 + cos(x3)
            # This is "vectorized" (fast) for one dataset, but we have to loop 10,000 times
            term1 = np.sin(X[:, 0] * coeffs[0])
            term2 = X[:, 1] * coeffs[1]
            term3 = np.cos(X[:, 2] * coeffs[2])
            y = term1 + term2 + term3
        else:
            # Simple Linear Path
            # y = x1*c1 + x2*c2 + x3*c3 - x4*c4
            y = (X[:, 0] * coeffs[0]) + (X[:, 1] * coeffs[1]) + \
                (X[:, 2] * coeffs[2]) - (X[:, 3] * coeffs[3])
                
        results_x.append(X)
        results_y.append(y)
        
    return results_x, results_y

def run_tests():
    # SETTINGS
    # 10,000 datasets * 256 rows = 2.5 Million rows of synthetic data
    N_DATASETS = 10_000 
    ROWS = 256
    
    print(f"--- SETTING UP: {N_DATASETS} Datasets on RTX 3080 ---")
    # Initialize Generator (Compilation happens here, so we don't time this)
    gen = GPUPriorGenerator(num_datasets=N_DATASETS, rows_per_dataset=ROWS)
    
    # --- TEST 1: SPEED BENCHMARK ---
    print("\n--- STARTING RACE: CPU vs GPU ---")
    
    # 1. GPU Run
    # We use perf_counter for high precision timing
    start_gpu = time.perf_counter()
    
    d_x, d_y = gen.generate()
    # We include the time to copy data back to RAM because that's 'fair'
    # (In a real pipeline, you'd keep it on GPU, making GPU even faster)
    h_x = d_x.copy_to_host() 
    h_y = d_y.copy_to_host()
    
    end_gpu = time.perf_counter()
    gpu_time = end_gpu - start_gpu
    print(f"GPU Time: {gpu_time:.4f} seconds (includes Memory Transfer)")
    
    # 2. CPU Run
    # We increase the CPU load to 2000 so the timer definitely catches it
    cpu_n = 2000
    print(f"Running CPU benchmark on subset ({cpu_n} datasets)...")
    
    start_cpu = time.perf_counter()
    cpu_version(cpu_n, ROWS)
    end_cpu = time.perf_counter()
    
    actual_cpu_time = end_cpu - start_cpu
    # Project how long 10,000 would have taken
    cpu_time_projected = actual_cpu_time * (N_DATASETS / cpu_n)
    
    print(f"CPU Time (Actual for {cpu_n}): {actual_cpu_time:.4f} seconds")
    print(f"CPU Time (Projected for full load): {cpu_time_projected:.4f} seconds")
    
    # Avoid division by zero
    if gpu_time > 0:
        speedup = cpu_time_projected / gpu_time
        print(f"\n>>> RESULT: RTX 3080 is {speedup:.1f}x FASTER than CPU <<<")
    else:
        print("GPU was instantaneous (check timer).")

    # --- TEST 2: DATA INTEGRITY ---
    print("\n--- VERIFYING DATA INTEGRITY ---")
    
    # Reshape flattened GPU output to inspect the first dataset
    first_dataset_x = h_x[0 : ROWS*4].reshape(ROWS, 4)
    first_dataset_y = h_y[0 : ROWS]
    
    # Sanity Check
    if np.all(first_dataset_y == 0):
        print("FAILED: Data is all zeros.")
    else:
        # Correlation Check
        model = LinearRegression()
        model.fit(first_dataset_x, first_dataset_y)
        score = model.score(first_dataset_x, first_dataset_y)
        
        print(f"R^2 Score of Dataset 0: {score:.4f}")
        if score > 0.8:
            print("PASSED: Linear relationship detected.")
        elif score > 0.05:
            print("PASSED: Non-Linear/Noisy relationship detected (Expected for Sin/Cos).")
        else:
            print("WARNING: No pattern found. (Might be pure noise).")

if __name__ == "__main__":
    run_tests()