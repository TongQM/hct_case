#!/usr/bin/env python3
"""
Test LBBD with a small number of blocks to debug infeasibility
"""

import sys
import os
import numpy as np
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib import GeoData
from lib.algorithm import Partition

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_small_lbbd():
    """Test LBBD with a small set of blocks"""
    
    # Small set of block groups for testing
    small_geoid_list = [
        "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", 
        "1500000US420034993002", "1500000US420034994001", "1500000US420034994002"
    ]
    
    bgfile_path = "2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"
    
    if not os.path.exists(bgfile_path):
        print(f"ERROR: Shapefile not found: {bgfile_path}")
        return False
    
    try:
        print("Setting up GeoData with 6 blocks...")
        bg_geodata = GeoData(bgfile_path, small_geoid_list, level='block_group')
        print(f"✓ GeoData initialized with {len(bg_geodata.short_geoid_list)} block groups")
        
        # Check connectivity
        import networkx as nx
        if nx.is_connected(bg_geodata.G):
            print("✓ Graph is connected")
        else:
            components = list(nx.connected_components(bg_geodata.G))
            print(f"⚠️  Graph has {len(components)} connected components")
            for i, comp in enumerate(components):
                print(f"    Component {i+1}: {comp}")
        
        # Create probability dictionary
        np.random.seed(42)
        probability_dict = {}
        total_demand = 0
        for block_id in bg_geodata.short_geoid_list:
            demand = max(10, np.random.poisson(50))
            total_demand += demand
            
        for block_id in bg_geodata.short_geoid_list:
            demand = max(10, np.random.poisson(50))
            probability_dict[block_id] = demand / total_demand
        
        print(f"✓ Created probability dict with total probability: {sum(probability_dict.values()):.4f}")
        
        # Test partition
        print("\nTesting Partition initialization...")
        partition = Partition(bg_geodata, num_districts=2, prob_dict=probability_dict, epsilon=0.1)
        print("✓ Partition initialized successfully")
        
        # Test LBBD
        print("\nRunning LBBD with 2 districts...")
        try:
            best_partition, best_cost, history = partition.benders_decomposition(
                max_iterations=3, tolerance=1e-2, max_cuts=5, verbose=True
            )
            print(f"✓ LBBD completed successfully!")
            print(f"  Best cost: {best_cost:.4f}")
            print(f"  Iterations: {len(history)}")
            return True
            
        except Exception as e:
            print(f"✗ LBBD failed: {e}")
            import traceback
            traceback.print_exc()
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_lbbd()
    sys.exit(0 if success else 1) 