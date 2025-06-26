#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from data import GeoData
import gurobipy as gp
from gurobipy import GRB
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_assignment_only():
    """Test with only assignment constraints, no flow constraints"""
    
    # Load data with small subset - use valid GEOIDs from config
    geoid_list = [
        "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", 
        "1500000US420034993002", "1500000US420034994001", "1500000US420034994002"
    ]
    
    print("Loading GeoData...")
    geodata = GeoData('./2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp', geoid_list, level='block_group')
    
    print(f"Loaded {len(geodata.short_geoid_list)} blocks")
    
    # Create a small master problem with only assignment constraints
    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    k = 2  # 2 districts
    
    print(f"\nCreating test assignment problem with {N} blocks, {k} districts")
    
    master = gp.Model("test_assignment_only")
    master.setParam('OutputFlag', 1)
    
    # Create assignment variables
    z = master.addVars(N, N, vtype=GRB.BINARY, name="z")
    
    # Basic assignment constraints
    for j in range(N):
        master.addConstr(gp.quicksum(z[i, j] for i in range(N)) == 1, name=f"assign_{j}")
    master.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name="num_districts")
    for i in range(N):
        for j in range(N):
            master.addConstr(z[j, i] <= z[i, i], name=f"valid_{i}_{j}")
    
    print(f"Added basic assignment constraints")
    
    # Try to solve
    master.setObjective(0, GRB.MINIMIZE)  # Dummy objective
    print("\nAttempting to solve...")
    master.optimize()
    
    print(f"Optimization status: {master.status}")
    if master.status == GRB.INFEASIBLE:
        print("Model is infeasible - computing IIS...")
        master.computeIIS()
        master.write("test_assignment_only_infeasible.ilp")
        print("IIS written to test_assignment_only_infeasible.ilp")
    elif master.status == GRB.OPTIMAL:
        print("Model solved successfully!")
        
        # Check solution
        print("\nSolution analysis:")
        roots = [i for i in range(N) if round(z[i, i].X) == 1]
        print(f"Selected roots: {[block_ids[i] for i in roots]}")
        
        for root_idx in roots:
            assigned = [j for j in range(N) if round(z[root_idx, j].X) == 1]
            print(f"District {block_ids[root_idx]}: {len(assigned)} blocks assigned")
            print(f"  Blocks: {[block_ids[j] for j in assigned]}")

if __name__ == "__main__":
    test_assignment_only() 