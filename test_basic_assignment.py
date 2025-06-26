#!/usr/bin/env python3
"""
Test basic assignment constraints without flow-based contiguity
"""

import sys
import os
import numpy as np
import logging
import gurobipy as gp
from gurobipy import GRB

sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from lib import GeoData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_basic_assignment():
    """Test basic assignment constraints only"""
    
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
        
        block_ids = bg_geodata.short_geoid_list
        N = len(block_ids)
        k = 2  # number of districts
        
        print(f"\nTesting basic assignment constraints only...")
        print(f"N = {N}, k = {k}")
        
        # Create basic assignment model
        model = gp.Model("basic_assignment")
        z = model.addVars(N, N, vtype=GRB.BINARY, name="z")
        
        # Basic assignment constraints
        # Each block assigned to one district
        for j in range(N):
            model.addConstr(gp.quicksum(z[i, j] for i in range(N)) == 1)
            
        # Exactly k districts
        model.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k)
        
        # Can only assign to a district if it's a root
        for i in range(N):
            for j in range(N):
                model.addConstr(z[j, i] <= z[i, i])
        
        # Simple objective: minimize sum of z variables
        model.setObjective(gp.quicksum(z[i, j] for i in range(N) for j in range(N)), GRB.MINIMIZE)
        
        model.setParam('OutputFlag', 1)
        model.optimize()
        
        if model.status == GRB.OPTIMAL:
            print(f"✓ Basic assignment model is feasible!")
            print(f"  Objective value: {model.objVal}")
            
            # Show solution
            print(f"\nSolution:")
            roots = []
            for i in range(N):
                if z[i, i].X > 0.5:
                    roots.append(i)
                    assigned = [j for j in range(N) if z[i, j].X > 0.5]
                    print(f"  District {i} (root {block_ids[i]}): {[block_ids[j] for j in assigned]}")
            
            return True
        else:
            print(f"✗ Basic assignment model failed with status: {model.status}")
            return False
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_assignment()
    sys.exit(0 if success else 1) 