#!/usr/bin/env python3
"""
Test script to validate GeoData arc structures
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib import GeoData
import numpy as np

def test_geodata_arcs():
    """Test GeoData arc structure construction"""
    
    # Use the same configuration as in the main script
    bgfile_path = "2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"
    bg_geoid_list = [
        "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", 
        "1500000US420034993002", "1500000US420034994001", "1500000US420034994002",
        "1500000US420034994003", "1500000US420034995001", "1500000US420034995002",
        "1500000US420034995003", "1500000US420034996001", "1500000US420034996002",
        "1500000US420034996003", "1500000US420034997001", "1500000US420034997002",
        "1500000US420034997003", "1500000US420034998001", "1500000US420034998002",
        "1500000US420034998003", "1500000US420034999001", "1500000US420034999002",
        "1500000US420034999003", "1500000US420035001001", "1500000US420035001002",
        "1500000US420035001003", "1500000US420035002001", "1500000US420035002002",
        "1500000US420035003001", "1500000US420035003002", "1500000US420035003003"
    ]
    
    if not os.path.exists(bgfile_path):
        print(f"ERROR: Shapefile not found: {bgfile_path}")
        return False
    
    try:
        print("Initializing GeoData...")
        bg_geodata = GeoData(bgfile_path, bg_geoid_list, level='block_group')
        print(f"✓ GeoData initialized with {len(bg_geodata.short_geoid_list)} block groups")
        
        # Test arc structures
        print("\nTesting arc structures...")
        
        # Check if arc_list exists
        if hasattr(bg_geodata, 'arc_list'):
            arc_list = bg_geodata.get_arc_list()
            print(f"✓ Arc list exists with {len(arc_list)} directed arcs")
        else:
            print("✗ Arc list not found in GeoData")
            return False
        
        # Check graph connectivity
        print(f"✓ Graph has {bg_geodata.G.number_of_nodes()} nodes and {bg_geodata.G.number_of_edges()} edges")
        
        # Test arc methods for each block
        print("\nTesting arc methods for each block...")
        blocks_with_no_arcs = []
        for block_id in bg_geodata.short_geoid_list:
            in_arcs = bg_geodata.get_in_arcs(block_id)
            out_arcs = bg_geodata.get_out_arcs(block_id)
            
            if not in_arcs and not out_arcs:
                blocks_with_no_arcs.append(block_id)
        
        if blocks_with_no_arcs:
            print(f"⚠️  {len(blocks_with_no_arcs)} blocks have no arcs: {blocks_with_no_arcs}")
            print("This might indicate isolated nodes in the graph")
        else:
            print("✓ All blocks have arc information")
        
        # Test a few specific blocks
        print("\nTesting specific blocks:")
        for i, block_id in enumerate(bg_geodata.short_geoid_list[:3]):
            in_arcs = bg_geodata.get_in_arcs(block_id)
            out_arcs = bg_geodata.get_out_arcs(block_id)
            print(f"  Block {block_id}: {len(in_arcs)} in-arcs, {len(out_arcs)} out-arcs")
        
        # Check if graph is connected
        import networkx as nx
        if nx.is_connected(bg_geodata.G):
            print("✓ Graph is connected")
        else:
            components = list(nx.connected_components(bg_geodata.G))
            print(f"⚠️  Graph has {len(components)} connected components")
            for i, comp in enumerate(components):
                print(f"    Component {i+1}: {len(comp)} nodes")
        
        print("\n" + "="*50)
        print("Arc structure test completed successfully!")
        return True
        
    except Exception as e:
        print(f"ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_geodata_arcs()
    sys.exit(0 if success else 1) 