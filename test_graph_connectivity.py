#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from data import GeoData
import networkx as nx
import logging
from lbbd_config import BG_GEOID_LIST, DATA_PATHS

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_graph_connectivity():
    """Test graph connectivity and structure for the full dataset"""
    
    print("Loading full GeoData with 130 blocks...")
    geodata = GeoData(DATA_PATHS['shapefile'], BG_GEOID_LIST, level='block_group')
    
    print(f"✓ Loaded {len(geodata.short_geoid_list)} blocks")
    print(f"✓ Graph has {geodata.G.number_of_nodes()} nodes and {geodata.G.number_of_edges()} edges")
    print(f"✓ Arc list has {len(geodata.get_arc_list())} directed arcs")
    
    # Check connectivity
    is_connected = nx.is_connected(geodata.G)
    print(f"✓ Graph is connected: {is_connected}")
    
    if not is_connected:
        # Find connected components
        components = list(nx.connected_components(geodata.G))
        print(f"✗ Graph has {len(components)} connected components:")
        for i, comp in enumerate(components):
            print(f"  Component {i+1}: {len(comp)} nodes")
            if len(comp) <= 5:  # Show small components
                print(f"    Nodes: {list(comp)}")
    
    # Check node degrees
    degrees = dict(geodata.G.degree())
    min_degree = min(degrees.values())
    max_degree = max(degrees.values())
    avg_degree = sum(degrees.values()) / len(degrees)
    
    print(f"✓ Node degrees - Min: {min_degree}, Max: {max_degree}, Avg: {avg_degree:.2f}")
    
    # Find nodes with very low degrees (potential issues)
    low_degree_nodes = [node for node, deg in degrees.items() if deg <= 1]
    if low_degree_nodes:
        print(f"⚠ Nodes with degree ≤ 1: {len(low_degree_nodes)}")
        print(f"  Examples: {low_degree_nodes[:5]}")
    
    # Check arc structure consistency
    print("\nChecking arc structure consistency...")
    missing_arcs = []
    for block_id in geodata.short_geoid_list:
        in_arcs = geodata.get_in_arcs(block_id)
        out_arcs = geodata.get_out_arcs(block_id)
        neighbors = list(geodata.G.neighbors(block_id))
        
        expected_in = len(neighbors)
        expected_out = len(neighbors)
        
        if len(in_arcs) != expected_in or len(out_arcs) != expected_out:
            missing_arcs.append((block_id, len(in_arcs), expected_in, len(out_arcs), expected_out))
    
    if missing_arcs:
        print(f"✗ Arc inconsistencies found: {len(missing_arcs)} nodes")
        for block_id, in_actual, in_expected, out_actual, out_expected in missing_arcs[:5]:
            print(f"  {block_id}: in={in_actual}/{in_expected}, out={out_actual}/{out_expected}")
    else:
        print("✓ Arc structures are consistent")
    
    # Test a smaller subset to see if that works
    print(f"\nTesting smaller subsets...")
    subset_sizes = [10, 20, 30]
    
    for size in subset_sizes:
        subset_geoids = BG_GEOID_LIST[:size]
        try:
            subset_geodata = GeoData(DATA_PATHS['shapefile'], subset_geoids, level='block_group')
            subset_connected = nx.is_connected(subset_geodata.G)
            print(f"✓ Subset of {size} blocks: {subset_geodata.G.number_of_nodes()} nodes, "
                  f"{subset_geodata.G.number_of_edges()} edges, connected: {subset_connected}")
        except Exception as e:
            print(f"✗ Subset of {size} blocks failed: {e}")

if __name__ == "__main__":
    test_graph_connectivity() 