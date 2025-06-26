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

def test_benders_balance():
    """Test why Benders decomposition creates imbalanced partitions"""
    
    # Use the same blocks as in the working tiny_test configuration
    block_geoids = ['4200300' + str(i).zfill(4) for i in range(1001, 1013)]  # 12 blocks
    
    geodata = GeoData(
        filepath='2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp',
        geoid_list=block_geoids,
        level='block_group'
    )
    
    print(f"Loaded {len(geodata.short_geoid_list)} blocks: {geodata.short_geoid_list}")
    
    if len(geodata.short_geoid_list) < 6:
        print("Not enough blocks loaded, using available blocks")
        # Use first 6 available blocks from the actual data
        all_blocks = geodata.short_geoid_list[:6]
    else:
        all_blocks = geodata.short_geoid_list[:6]  # Use first 6 for testing
    
    print(f"Testing with {len(all_blocks)} blocks: {all_blocks}")
    
    # Create simple uniform probability distribution
    prob_dict = {block: 1.0/len(all_blocks) for block in all_blocks}
    
    # Test with 2 districts
    partition = Partition(geodata, num_districts=2, prob_dict=prob_dict, epsilon=0.001)
    partition.short_geoid_list = all_blocks  # Override for testing
    
    print("\n=== Testing Master Problem (No Benders Cuts) ===")
    test_master_problem_balance(all_blocks, geodata, prob_dict)
    
    print("\n=== Testing One Iteration of Benders ===")
    test_benders_iteration(all_blocks, geodata, prob_dict)

def test_master_problem_balance(block_ids, geodata, prob_dict):
    """Test the master problem with balanced objective"""
    N = len(block_ids)
    k = 2  # districts
    
    # Get arc structures
    arc_list = geodata.get_arc_list()
    print(f"Using {len(arc_list)} directed arcs")
    
    model = gp.Model("master_balanced")
    model.setParam('OutputFlag', 1)
    
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
    
    # BALANCED objective: minimize sum of district sizes squared
    district_sizes = [gp.quicksum(z[j, i] for j in range(N)) for i in range(N)]
    model.setObjective(gp.quicksum(size * size for size in district_sizes), GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("✓ Master problem with balanced objective solved successfully")
        
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
        return True
        
    else:
        print(f"✗ Master problem failed. Status: {model.status}")
        return False

def test_benders_iteration(block_ids, geodata, prob_dict):
    """Test one iteration of Benders decomposition to see what cuts are generated"""
    N = len(block_ids)
    k = 2
    
    # Create partition object
    partition = Partition(geodata, num_districts=k, prob_dict=prob_dict, epsilon=0.001)
    partition.short_geoid_list = block_ids
    
    print("Running one Benders iteration...")
    
    # Get arc structures
    arc_list = geodata.get_arc_list()
    
    # Create master problem (first iteration - no cuts)
    master = gp.Model("benders_master")
    master.setParam('OutputFlag', 0)
    
    z = master.addVars(N, N, vtype=GRB.BINARY, name="assignment")
    o = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="objval")
    f = master.addVars(block_ids, arc_list, lb=0.0, vtype=GRB.CONTINUOUS, name="flows")
    
    # Basic assignment constraints
    master.addConstrs((gp.quicksum(z[j, i] for i in range(N)) == 1 for j in range(N)), name='one_assignment')
    master.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name='num_districts')
    master.addConstrs((z[j, i] <= z[i, i] for i in range(N) for j in range(N)), name='validity')
    
    # Flow constraints
    for i, root_id in enumerate(block_ids):
        master.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(root_id)) == 0, name=f"no_return_flow_{root_id}")
        for j, block_id in enumerate(block_ids):
            if block_id != root_id:
                master.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(block_id)) - gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_out_arcs(block_id)) == z[j, i], name=f"flow_assign_{block_id}_{root_id}")
                master.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(block_id)) <= (N - 1) * z[j, i], name=f"flow_restrict_{block_id}_{root_id}")
    
    # Benders objective: minimize o (no cuts yet, so o can be 0)
    master.setObjective(o, GRB.MINIMIZE)
    
    master.optimize()
    
    if master.status != GRB.OPTIMAL:
        print(f"✗ Master problem failed. Status: {master.status}")
        return
    
    print("✓ Master problem solved")
    
    # Get solution
    z_sol = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            z_sol[j, i] = z[j, i].X
    
    o_sol = o.X
    print(f"Master objective value: {o_sol}")
    
    # Find roots and assignments
    roots = [i for i in range(N) if round(z_sol[i, i]) == 1]
    print(f"Found {len(roots)} districts with roots: {[block_ids[i] for i in roots]}")
    
    district_costs = []
    for root in roots:
        assigned_blocks = [j for j in range(N) if round(z_sol[j, root]) == 1]
        assigned_block_ids = [block_ids[j] for j in assigned_blocks]
        print(f"District {block_ids[root]}: {len(assigned_blocks)} blocks = {assigned_block_ids}")
        
        # Solve SDP for this district
        cost, x_star, subgrad, T_star, alpha_i = partition._SDP_benders(
            assigned_block_ids, block_ids[root], prob_dict, partition.epsilon)
        district_costs.append((cost, root, assigned_blocks, subgrad, x_star, T_star, alpha_i))
        print(f"  District cost: {cost:.4f}, alpha: {alpha_i:.4f}, T*: {T_star:.4f}")
    
    # Find worst district
    worst_idx = int(np.argmax([c[0] for c in district_costs]))
    worst_cost, worst_root, worst_blocks, worst_subgrad, x_star, T_star, alpha_i = district_costs[worst_idx]
    
    print(f"\nWorst district: {block_ids[worst_root]} with cost {worst_cost:.4f}")
    print(f"This will generate a Benders cut")
    
    # The issue: Benders minimizes the WORST-CASE district cost
    # This can lead to very imbalanced solutions if one district handles most blocks efficiently
    print(f"\n🔍 INSIGHT: Benders minimizes worst-case cost = {worst_cost:.4f}")
    print(f"But this doesn't consider balance! Large districts might be 'efficient' in worst-case terms.")

if __name__ == "__main__":
    test_benders_balance() 