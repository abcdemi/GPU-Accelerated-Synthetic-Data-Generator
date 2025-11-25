import time
import numpy as np
from gpu_priors import GPUPriorGenerator
from sklearn.linear_model import LinearRegression

def cpu_version(num_datasets, rows):
    """Slow CPU simulation for comparison"""
    results_x = []
    results_y = []
    for _ in range(num_datasets):
        # Generate data
        X = np.random.rand(rows, 4).astype(np.float32)
        coeffs = np.random.randn(4)
        # Simple Linear
        y = np.dot(X, coeffs)
        results_x.append(X)
        results_y.append(y)
    return results_x, results_y

def run_tests():
    # SETTINGS
    # Let's generate 10,000 datasets! 
    # Each with 256 rows. Total 2.5 Million rows generated.
    N_DATASETS = 10_000 
    ROWS = 256
    
    print(f"--- SETTING UP: {N_DATASETS} Datasets on RTX 3080 ---")
    gen = GPUPriorGenerator(num_datasets=N_DATASETS, rows_per_dataset=ROWS)
    
    # --- TEST 1: SPEED BENCHMARK ---
    print("\n--- STARTING RACE: CPU vs GPU ---")
    
    # GPU Run
    start_gpu = time.time()
    d_x, d_y = gen.generate()
    # Force Copy back to CPU to include transfer overhead in the test
    h_x = d_x.copy_to_host() 
    h_y = d_y.copy_to_host()
    end_gpu = time.time()
    gpu_time = end_gpu - start_gpu
    print(f"GPU Time: {gpu_time:.4f} seconds")
    
    # CPU Run (We will reduce N for CPU because it's too slow)
    cpu_n = 500
    print(f"Running CPU benchmark on smaller subset ({cpu_n} datasets)...")
    start_cpu = time.time()
    cpu_version(cpu_n, ROWS)
    end_cpu = time.time()
    cpu_time_projected = (end_cpu - start_cpu) * (N_DATASETS / cpu_n)
    
    print(f"CPU Time (Projected for full load): {cpu_time_projected:.4f} seconds")
    print(f"SPEEDUP FACTOR: {cpu_time_projected / gpu_time:.1f}x FASTER")

    # --- TEST 2: DATA INTEGRITY (Does the data make sense?) ---
    print("\n--- VERIFYING DATA INTEGRITY ---")
    
    # Let's reshape the GPU output to inspect the first dataset
    # Recall X was flattened: [Total_Rows * 4]
    first_dataset_x = h_x[0 : ROWS*4].reshape(ROWS, 4)
    first_dataset_y = h_y[0 : ROWS]
    
    # Sanity Check: Are they all zeros?
    if np.all(first_dataset_y == 0):
        print("FAILED: Data is all zeros.")
    else:
        print("PASSED: Data contains non-zero values.")
        
    # Correlation Check:
    # Since we generated Y based on X, a Linear Regression should find a fit.
    # Note: If the kernel used the Sine wave (Non-linear) path, score might be lower, but still positive.
    model = LinearRegression()
    model.fit(first_dataset_x, first_dataset_y)
    score = model.score(first_dataset_x, first_dataset_y)
    
    print(f"R^2 Score of Dataset 0: {score:.4f}")
    if score > 0.8:
        print("PASSED: Strong Causal Relationship detected (Linear).")
    elif score > 0.1:
        print("PASSED: Weak/Non-Linear Relationship detected.")
    else:
        print("WARNING: No relationship found (Check logic).")

if __name__ == "__main__":
    run_tests()