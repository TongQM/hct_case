#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'lib'))

from data import GeoData
import gurobipy as gp
from gurobipy import GRB
import logging
from lbbd_config import BG_GEOID_LIST, DATA_PATHS

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_assignment_only_large():
    """Test assignment constraints without flow constraints on full dataset"""
    
    print("Loading full GeoData with 130 blocks...")
    geodata = GeoData(DATA_PATHS['shapefile'], BG_GEOID_LIST, level='block_group')
    
    print(f"✓ Loaded {len(geodata.short_geoid_list)} blocks")
    
    # Create assignment problem
    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    k = 3  # 3 districts as in the config
    
    print(f"\nCreating assignment problem with {N} blocks, {k} districts")
    
    master = gp.Model("assignment_only_large")
    master.setParam('OutputFlag', 1)
    
    # Create assignment variables
    z = master.addVars(N, N, vtype=GRB.BINARY, name="z")
    print(f"Created {N*N} assignment variables")
    
    # Basic assignment constraints
    for j in range(N):
        master.addConstr(gp.quicksum(z[i, j] for i in range(N)) == 1, name=f"assign_{j}")
    master.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name="num_districts")
    for i in range(N):
        for j in range(N):
            if i != j:
                master.addConstr(z[i, j] <= z[i, i], name=f"valid_{i}_{j}")
    
    print(f"Added assignment constraints")
    print(f"Model has {master.NumVars} variables and {master.NumConstrs} constraints")
    
    # Try to solve
    master.setObjective(0, GRB.MINIMIZE)  # Dummy objective
    print("\nAttempting to solve...")
    master.optimize()
    
    print(f"Optimization status: {master.status}")
    if master.status == GRB.INFEASIBLE:
        print("✗ Assignment model is infeasible")
        master.computeIIS()
        master.write("assignment_only_large_infeasible.ilp")
        print("IIS written to assignment_only_large_infeasible.ilp")
    elif master.status == GRB.OPTIMAL:
        print("✓ Assignment model solved successfully!")
        
        # Check solution
        print("\nSolution analysis:")
        roots = [i for i in range(N) if round(z[i, i].X) == 1]
        print(f"Selected roots: {[block_ids[i] for i in roots]}")
        
        for root_idx in roots:
            assigned = [j for j in range(N) if round(z[root_idx, j].X) == 1]
            print(f"District {block_ids[root_idx]}: {len(assigned)} blocks assigned")
        
        # Check if districts are contiguous (just for information)
        print(f"\nChecking contiguity (informational):")
        for root_idx in roots:
            assigned = [j for j in range(N) if round(z[root_idx, j].X) == 1]
            assigned_ids = [block_ids[j] for j in assigned]
            
            # Create subgraph of assigned blocks
            subgraph = geodata.G.subgraph(assigned_ids)
            is_connected = len(assigned_ids) == 1 or (len(assigned_ids) > 1 and 
                          len(list(nx.connected_components(subgraph))) == 1)
            
            print(f"  District {block_ids[root_idx]}: {len(assigned)} blocks, connected: {is_connected}")
    
    else:
        print(f"✗ Optimization failed with status {master.status}")

if __name__ == "__main__":
    import networkx as nx
    test_assignment_only_large() 