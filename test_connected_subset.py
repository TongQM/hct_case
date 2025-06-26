#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from data import GeoData
from algorithm import Partition
import networkx as nx
import logging
import numpy as np
from lbbd_config import BG_GEOID_LIST, DATA_PATHS

# Set up logging
logging.basicConfig(level=logging.INFO)

def find_connected_subset(geodata, target_size=20):
    """Find a connected subset of nodes of approximately target_size"""
    
    # Start with a random node and grow the connected component
    start_node = geodata.short_geoid_list[0]
    visited = {start_node}
    queue = [start_node]
    
    while len(visited) < target_size and queue:
        current = queue.pop(0)
        neighbors = list(geodata.G.neighbors(current))
        
        for neighbor in neighbors:
            if neighbor not in visited and len(visited) < target_size:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return list(visited)

def test_connected_subset():
    """Test LBBD with a connected subset"""
    
    print("Loading full GeoData...")
    full_geodata = GeoData(DATA_PATHS['shapefile'], BG_GEOID_LIST, level='block_group')
    
    # Find a connected subset
    target_size = 20
    connected_subset = find_connected_subset(full_geodata, target_size)
    print(f"Found connected subset of {len(connected_subset)} blocks")
    
    # Create corresponding full GEOIDs
    subset_full_geoids = []
    for short_id in connected_subset:
        # Find the full GEOID that corresponds to this short ID
        for full_geoid in BG_GEOID_LIST:
            if full_geoid.endswith(short_id):
                subset_full_geoids.append(full_geoid)
                break
    
    print(f"Mapped to {len(subset_full_geoids)} full GEOIDs")
    
    # Create GeoData for the subset
    print("Creating GeoData for connected subset...")
    subset_geodata = GeoData(DATA_PATHS['shapefile'], subset_full_geoids, level='block_group')
    
    # Verify connectivity
    is_connected = nx.is_connected(subset_geodata.G)
    print(f"✓ Subset is connected: {is_connected}")
    print(f"✓ Subset has {subset_geodata.G.number_of_nodes()} nodes and {subset_geodata.G.number_of_edges()} edges")
    
    if not is_connected:
        print("✗ Subset is not connected, cannot test LBBD")
        return
    
    # Create probability dict
    prob_dict = {}
    total_prob = 0.0
    for block_id in subset_geodata.short_geoid_list:
        # Simple uniform probability for testing
        prob = 1.0 / len(subset_geodata.short_geoid_list)
        prob_dict[block_id] = prob
        total_prob += prob
    
    print(f"✓ Created probability dict with total probability: {total_prob:.4f}")
    
    # Test LBBD
    print(f"\nTesting LBBD with {len(subset_geodata.short_geoid_list)} blocks and 2 districts...")
    
    try:
        partition = Partition(subset_geodata, num_districts=2, prob_dict=prob_dict, epsilon=0.1)
        print("✓ Partition initialized successfully")
        
        # Run LBBD with minimal settings
        best_partition, best_cost, history = partition.benders_decomposition(
            max_iterations=3,
            tolerance=1e-2,
            max_cuts=10,
            verbose=True
        )
        
        print(f"✓ LBBD completed successfully!")
        print(f"  Best cost: {best_cost:.4f}")
        print(f"  Iterations: {len(history)}")
        
    except Exception as e:
        print(f"✗ LBBD failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_connected_subset() 