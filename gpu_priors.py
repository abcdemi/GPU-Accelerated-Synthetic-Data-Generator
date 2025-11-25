import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math

# --- THE CUDA KERNEL ---
# This runs on the RTX 3080.
# Logic: Each BLOCK generates one distinct "Dataset". 
#        Each THREAD generates one "Row" of data.
@cuda.jit
def generate_tabular_priors_kernel(rng_states, out_x, out_y, coeffs, complexities):
    # Calculate global thread ID for RNG
    tid = cuda.grid(1)
    
    # Calculate logical indices
    # blockIdx.x = Which "Universe" (Dataset) are we in?
    # threadIdx.x = Which Row are we generating?
    dataset_idx = cuda.blockIdx.x
    row_idx = cuda.threadIdx.x
    
    # FIX: Check bounds against out_y (Total Rows), not out_x shape
    if tid < out_y.shape[0]:
        # 1. Generate Features (X)
        # We have 4 features per dataset
        x1 = xoroshiro128p_uniform_float32(rng_states, tid)
        x2 = xoroshiro128p_uniform_float32(rng_states, tid)
        x3 = xoroshiro128p_uniform_float32(rng_states, tid)
        x4 = xoroshiro128p_uniform_float32(rng_states, tid)
        
        # Store X in global memory (flattened)
        # Structure: [Dataset 0 Row 0, Dataset 0 Row 1... ]
        # Each thread writes 4 values, so we multiply tid by 4
        base_idx = tid * 4
        out_x[base_idx + 0] = x1
        out_x[base_idx + 1] = x2
        out_x[base_idx + 2] = x3
        out_x[base_idx + 3] = x4
        
        # 2. Generate Label (y) based on a "Prior" (Equation)
        # Every dataset (block) has its own random coefficients passed in via 'coeffs'
        c1 = coeffs[dataset_idx, 0]
        c2 = coeffs[dataset_idx, 1]
        c3 = coeffs[dataset_idx, 2]
        c4 = coeffs[dataset_idx, 3]
        
        # 3. Apply Complexity (Non-linearity)
        # If 'complexities' for this dataset is > 0.5, use Sin, else Linear
        complexity = complexities[dataset_idx]
        
        y_val = 0.0
        if complexity > 0.5:
            # A non-linear relationship (e.g., Sine wave)
            y_val = math.sin(x1 * c1) + (x2 * c2) + math.cos(x3 * c3)
        else:
            # A simple linear relationship
            y_val = (x1 * c1) + (x2 * c2) + (x3 * c3) - (x4 * c4)
            
        # Write Y to global memory
        out_y[tid] = y_val

class GPUPriorGenerator:
    def __init__(self, num_datasets=1000, rows_per_dataset=256):
        self.num_datasets = num_datasets
        self.rows = rows_per_dataset
        self.total_threads = num_datasets * rows_per_dataset
        
        # Init RNG states on GPU (Expensive, do once)
        self.rng_states = create_xoroshiro128p_states(self.total_threads, seed=42)
        
    def generate(self):
        # 1. Prepare Random Coefficients for each dataset (The "Rules" of the universe)
        # We generate these on CPU and move to GPU because it's small data
        h_coeffs = np.random.randn(self.num_datasets, 4).astype(np.float32)
        h_complexities = np.random.rand(self.num_datasets).astype(np.float32)
        
        d_coeffs = cuda.to_device(h_coeffs)
        d_complexities = cuda.to_device(h_complexities)
        
        # 2. Allocate output memory on GPU
        # 4 floats for X, 1 float for Y per row
        d_out_x = cuda.device_array(self.total_threads * 4, dtype=np.float32)
        d_out_y = cuda.device_array(self.total_threads, dtype=np.float32)
        
        # 3. Launch Kernel
        # Blocks = Datasets, Threads = Rows
        threads_per_block = self.rows
        blocks = self.num_datasets
        
        generate_tabular_priors_kernel[blocks, threads_per_block](
            self.rng_states, d_out_x, d_out_y, d_coeffs, d_complexities
        )
        cuda.synchronize() # Wait for RTX 3080 to finish
        
        return d_out_x, d_out_y