#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import gurobipy as gp
from gurobipy import GRB
import numpy as np
from lib.data import GeoData
from lib.algorithm import Partition
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_partition_balance():
    """Test why partitions are becoming imbalanced"""
    
    # Load data using correct constructor
    block_geoids = [
        '420030001001', '420030001002', '420030002001', '420030002002', 
        '420030003001', '420030003002', '420030004001', '420030004002',
        '420030005001', '420030005002', '420030006001', '420030006002'
    ]
    
    geodata = GeoData(
        filepath='2023_shape_files/tl_2023_42_bg.shp',
        geoid_list=block_geoids,
        level='block_group'
    )
    
    print(f"Loaded {len(geodata.short_geoid_list)} blocks: {geodata.short_geoid_list}")
    
    # Create simple uniform probability distribution
    prob_dict = {block: 1.0/len(geodata.short_geoid_list) for block in geodata.short_geoid_list}
    
    # Test with 3 districts
    partition = Partition(geodata, num_districts=3, prob_dict=prob_dict, epsilon=0.001)
    
    # Run a simple test to see the assignment pattern
    print("\n=== Testing Assignment Without Flow Constraints ===")
    test_assignment_only(geodata, prob_dict)
    
    print("\n=== Testing Assignment With Flow Constraints ===")
    test_assignment_with_flow(geodata, prob_dict)

def test_assignment_only(geodata, prob_dict):
    """Test assignment constraints only (no flow)"""
    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    k = 3  # districts
    
    model = gp.Model("assignment_only")
    model.setParam('OutputFlag', 0)
    
    # Variables
    z = model.addVars(N, N, vtype=GRB.BINARY, name="assignment")
    
    # Basic assignment constraints
    model.addConstrs((gp.quicksum(z[j, i] for i in range(N)) == 1 for j in range(N)), name='one_assignment')
    model.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name='num_districts')
    model.addConstrs((z[j, i] <= z[i, i] for i in range(N) for j in range(N)), name='validity')
    
    # Simple objective: minimize sum of district sizes squared (encourage balance)
    district_sizes = [gp.quicksum(z[j, i] for j in range(N)) for i in range(N)]
    model.setObjective(gp.quicksum(size * size for size in district_sizes), GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("✓ Assignment-only model solved successfully")
        
        # Check solution
        roots = []
        for i in range(N):
            if round(z[i, i].X) == 1:
                roots.append(i)
        
        print(f"Found {len(roots)} districts with roots: {[block_ids[i] for i in roots]}")
        
        # Check district assignments and sizes
        district_sizes = []
        for root_idx in roots:
            assigned = []
            for j in range(N):
                if round(z[j, root_idx].X) == 1:
                    assigned.append(block_ids[j])
            district_sizes.append(len(assigned))
            print(f"District {block_ids[root_idx]}: {len(assigned)} blocks = {assigned}")
        
        print(f"District sizes: {district_sizes} (balanced: {max(district_sizes) - min(district_sizes) <= 1})")
        
    else:
        print(f"✗ Assignment-only model failed. Status: {model.status}")

def test_assignment_with_flow(geodata, prob_dict):
    """Test assignment with flow constraints"""
    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    k = 3  # districts
    
    # Get arc structures
    arc_list = geodata.get_arc_list()
    print(f"Using {len(arc_list)} directed arcs")
    
    model = gp.Model("assignment_with_flow")
    model.setParam('OutputFlag', 0)
    
    # Variables
    z = model.addVars(N, N, vtype=GRB.BINARY, name="assignment")
    f = model.addVars(block_ids, arc_list, lb=0.0, vtype=GRB.CONTINUOUS, name="flows")
    
    # Basic assignment constraints
    model.addConstrs((gp.quicksum(z[j, i] for i in range(N)) == 1 for j in range(N)), name='one_assignment')
    model.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name='num_districts')
    model.addConstrs((z[j, i] <= z[i, i] for i in range(N) for j in range(N)), name='validity')
    
    # Flow constraints for contiguity
    for i, root_id in enumerate(block_ids):
        # Root has no inflow
        model.addConstr(
            gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(root_id)) == 0, 
            name=f"no_inflow_{root_id}"
        )
        
        for j, block_id in enumerate(block_ids):
            if block_id != root_id:
                # Flow conservation: inflow - outflow = z[j,i] 
                model.addConstr(
                    gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(block_id)) - 
                    gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_out_arcs(block_id)) == z[j, i], 
                    name=f"flow_conservation_{block_id}_{root_id}"
                )
                
                # Flow capacity
                model.addConstr(
                    gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(block_id)) <= (N-1) * z[j, i], 
                    name=f"flow_capacity_{block_id}_{root_id}"
                )
    
    # Simple objective: minimize sum of district sizes squared (encourage balance)
    district_sizes = [gp.quicksum(z[j, i] for j in range(N)) for i in range(N)]
    model.setObjective(gp.quicksum(size * size for size in district_sizes), GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("✓ Assignment+flow model solved successfully")
        
        # Check solution
        roots = []
        for i in range(N):
            if round(z[i, i].X) == 1:
                roots.append(i)
        
        print(f"Found {len(roots)} districts with roots: {[block_ids[i] for i in roots]}")
        
        # Check district assignments and sizes
        district_sizes = []
        for root_idx in roots:
            assigned = []
            for j in range(N):
                if round(z[j, root_idx].X) == 1:
                    assigned.append(block_ids[j])
            district_sizes.append(len(assigned))
            print(f"District {block_ids[root_idx]}: {len(assigned)} blocks = {assigned}")
        
        print(f"District sizes: {district_sizes} (balanced: {max(district_sizes) - min(district_sizes) <= 1})")
        
    else:
        print(f"✗ Assignment+flow model failed. Status: {model.status}")
        if model.status == GRB.INFEASIBLE:
            model.computeIIS()
            model.write("flow_debug_infeasible.ilp")
            print("IIS written to flow_debug_infeasible.ilp")

if __name__ == "__main__":
    test_partition_balance() 