#!/usr/bin/env python3
"""
Minimal debug test to identify where LBBD is hanging
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from lib.algorithm import Partition
from test_lbbd_final import ToyGeoData

def debug_lbbd():
    print("=== MINIMAL LBBD DEBUG TEST ===")
    
    # Very simple setup
    grid_size = 3  # Smaller than 4x4
    n_blocks = 9
    num_districts = 2  # Fewer districts
    
    print(f"Creating {grid_size}×{grid_size} grid with {num_districts} districts...")
    
    # Setup
    toy_geo = ToyGeoData(n_blocks, grid_size)
    toy_geo.beta = 0.7120
    toy_geo.Lambda = 25.0
    toy_geo.wr = 1.0
    toy_geo.wv = 10.0
    
    print("Creating probability distribution...")
    prob_dict = {}
    for i, block_id in enumerate(toy_geo.short_geoid_list):
        prob_dict[block_id] = 1.0 / n_blocks  # Uniform distribution
    
    print("Creating ODD features...")
    Omega_dict = {}
    for i, block_id in enumerate(toy_geo.short_geoid_list):
        Omega_dict[block_id] = np.array([2.0, 2.0])  # Constant ODD
    
    def J_function(omega_vector):
        return 0.5 * omega_vector[0] + 0.4 * omega_vector[1]
    
    print("Creating Partition instance...")
    partition = Partition(toy_geo, num_districts=num_districts, 
                         prob_dict=prob_dict, epsilon=0.2)
    
    print("Starting LBBD with minimal settings...")
    print("Max iterations: 3")
    print("Tolerance: 0.1 (loose)")
    print("Parallel: True")
    print("Epsilon scaling: Removed")
    
    try:
        result = partition.benders_decomposition(
            max_iterations=3,  # Very few iterations
            tolerance=0.1,     # Loose tolerance
            verbose=True,
            Omega_dict=Omega_dict,
            J_function=J_function,
            parallel=True     # Enable parallel
        )
        print("✅ LBBD completed successfully!")
        print(f"Result: {result['converged']}, Cost: {result['best_cost']:.4f}")
    except Exception as e:
        print(f"❌ LBBD failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_lbbd()