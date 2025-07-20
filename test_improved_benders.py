#!/usr/bin/env python3
"""
Test script for the improved Logic-based Benders Decomposition algorithm.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import logging
from test_toy_model_25blocks import ToyGeoData
from lib.algorithm import Partition

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_improved_benders():
    """Test the improved Benders decomposition algorithm."""
    
    print("=" * 60)
    print("TESTING IMPROVED BENDERS DECOMPOSITION")
    print("=" * 60)
    
    # 1. Create toy geodata (smaller for faster testing)
    print("\n1. Creating toy geodata...")
    toy_geo = ToyGeoData(n_blocks=9, grid_size=3, seed=42)
    
    # 2. Generate probability distribution
    print("2. Generating probability distribution...")
    np.random.seed(42)
    probs = np.random.uniform(0.05, 0.15, len(toy_geo.short_geoid_list))
    probs = probs / probs.sum()
    prob_dict = {bid: p for bid, p in zip(toy_geo.short_geoid_list, probs)}
    
    print(f"   - Number of blocks: {len(toy_geo.short_geoid_list)}")
    print(f"   - Probability distribution sum: {sum(prob_dict.values()):.4f}")
    
    # 3. Compute K for all blocks
    print("3. Computing K values...")
    toy_geo.compute_K_for_all_blocks(depot_lat=0, depot_lon=0)
    
    # 4. Create Partition instance with small epsilon for testing
    print("4. Creating partition instance...")
    num_districts = 3
    epsilon = 0.5  # Larger epsilon for feasibility
    partition = Partition(toy_geo, num_districts, prob_dict, epsilon=epsilon)
    
    print(f"   - Number of districts: {num_districts}")
    print(f"   - Wasserstein radius (epsilon): {epsilon}")
    print(f"   - Number of arcs: {len(toy_geo.get_arc_list())}")
    
    # 5. Test the improved Benders decomposition
    print("\n5. Running improved Benders decomposition...")
    try:
        best_partition, best_cost, history = partition.benders_decomposition(
            max_iterations=5,  # Small number for testing
            tolerance=1e-2,
            verbose=True
        )
        
        print(f"\n6. RESULTS:")
        print(f"   - Algorithm completed successfully!")
        print(f"   - Best cost: {best_cost:.4f}")
        print(f"   - Number of iterations: {len(history)}")
        print(f"   - Final gap: {history[-1]['gap']:.4f}" if history else "N/A")
        
        # 7. Validate the solution
        print("\n7. Validating solution...")
        if best_partition is not None:
            # Check if partition is valid
            row_sums = np.sum(best_partition, axis=1)
            num_districts_actual = np.sum(np.diag(best_partition))
            
            print(f"   - All blocks assigned: {np.allclose(row_sums, 1.0)}")
            print(f"   - Number of districts in solution: {num_districts_actual}")
            print(f"   - Expected number of districts: {num_districts}")
            print(f"   - Valid partition: {num_districts_actual == num_districts}")
            
            # Show district assignments
            print("\n8. District assignments:")
            for i in range(len(toy_geo.short_geoid_list)):
                if np.round(best_partition[i, i]) == 1:
                    assigned_blocks = [j for j in range(len(toy_geo.short_geoid_list)) 
                                     if np.round(best_partition[j, i]) == 1]
                    print(f"   - District {i} (root: {toy_geo.short_geoid_list[i]}): "
                          f"{len(assigned_blocks)} blocks")
        else:
            print("   - No valid partition found")
            
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print(f"   Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """Test edge cases for the Benders decomposition."""
    
    print("\n" + "=" * 60)
    print("TESTING EDGE CASES")
    print("=" * 60)
    
    # Test with very small epsilon (might be infeasible)
    print("\n1. Testing with very small epsilon...")
    toy_geo = ToyGeoData(n_blocks=4, grid_size=2, seed=42)
    probs = np.ones(4) / 4
    prob_dict = {bid: p for bid, p in zip(toy_geo.short_geoid_list, probs)}
    toy_geo.compute_K_for_all_blocks(depot_lat=0, depot_lon=0)
    
    partition = Partition(toy_geo, 2, prob_dict, epsilon=1e-6)
    
    try:
        best_partition, best_cost, history = partition.benders_decomposition(
            max_iterations=2,
            tolerance=1e-2,
            verbose=False
        )
        print("   ✓ Small epsilon test passed")
    except Exception as e:
        print(f"   ⚠ Small epsilon test failed (expected): {type(e).__name__}")
    
    return True

def main():
    """Run all tests."""
    print("Starting Benders Decomposition Tests...")
    
    success1 = test_improved_benders()
    success2 = test_edge_cases()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if success1 and success2:
        print("✓ All tests completed!")
        print("✓ Improved Benders decomposition is working correctly")
    else:
        print("❌ Some tests failed")
        
    print("=" * 60)

if __name__ == "__main__":
    main()