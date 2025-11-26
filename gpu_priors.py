import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import math

@cuda.jit
def generate_tabular_priors_kernel(rng_states, out_x, out_y, coeffs, complexities):
    tid = cuda.grid(1)
    dataset_idx = cuda.blockIdx.x
    row_idx = cuda.threadIdx.x
    
    if tid < out_y.shape[0]:
        # 1. Generate 4 random features
        x1 = xoroshiro128p_uniform_float32(rng_states, tid) * 2.0 - 1.0 # Range -1 to 1
        x2 = xoroshiro128p_uniform_float32(rng_states, tid) * 2.0 - 1.0
        x3 = xoroshiro128p_uniform_float32(rng_states, tid) * 2.0 - 1.0
        x4 = xoroshiro128p_uniform_float32(rng_states, tid) * 2.0 - 1.0
        
        # Store X
        base_idx = tid * 4
        out_x[base_idx + 0] = x1
        out_x[base_idx + 1] = x2
        out_x[base_idx + 2] = x3
        out_x[base_idx + 3] = x4
        
        # 2. HEAVY MATH: Non-Linear Causal Chain
        c1 = coeffs[dataset_idx, 0]
        c2 = coeffs[dataset_idx, 1]
        c3 = coeffs[dataset_idx, 2]
        c4 = coeffs[dataset_idx, 3]
        
        # "Hard" calculation mimicking complex real-world data distributions
        # GPU eats this for breakfast. CPU struggles.
        term1 = math.sin(x1 * c1 * 3.14)
        term2 = math.cos(x2 * c2 * 3.14)
        term3 = math.tanh(x3 * c3)
        term4 = math.exp(x4 * c4 * -1.0) # Decay
        
        y_val = term1 + term2 + term3 + term4
            
        out_y[tid] = y_val

class GPUPriorGenerator:
    def __init__(self, num_datasets=1000, rows_per_dataset=256):
        self.num_datasets = num_datasets
        self.rows = rows_per_dataset
        self.total_threads = num_datasets * rows_per_dataset
        self.rng_states = create_xoroshiro128p_states(self.total_threads, seed=42)
        
    def generate(self):
        h_coeffs = np.random.randn(self.num_datasets, 4).astype(np.float32)
        # Unused now, but kept for compatibility
        h_complexities = np.random.rand(self.num_datasets).astype(np.float32) 
        
        d_coeffs = cuda.to_device(h_coeffs)
        d_complexities = cuda.to_device(h_complexities)
        
        d_out_x = cuda.device_array(self.total_threads * 4, dtype=np.float32)
        d_out_y = cuda.device_array(self.total_threads, dtype=np.float32)
        
        generate_tabular_priors_kernel[self.num_datasets, self.rows](
            self.rng_states, d_out_x, d_out_y, d_coeffs, d_complexities
        )
        cuda.synchronize()
        return d_out_x, d_out_y