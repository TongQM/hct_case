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

def test_flow_indexing():
    """Test the flow variable indexing to identify the issue"""
    
    # Load data with small subset - use valid GEOIDs from config
    geoid_list = [
        "1500000US420034980001", "1500000US420034980002", "1500000US420034993001", 
        "1500000US420034993002", "1500000US420034994001", "1500000US420034994002"
    ]
    
    print("Loading GeoData...")
    geodata = GeoData('./2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp', geoid_list, level='block_group')
    
    print(f"Loaded {len(geodata.short_geoid_list)} blocks")
    print(f"Arc list has {len(geodata.get_arc_list())} arcs")
    
    # Check arc structure
    for i, block_id in enumerate(geodata.short_geoid_list[:3]):
        in_arcs = geodata.get_in_arcs(block_id)
        out_arcs = geodata.get_out_arcs(block_id)
        print(f"Block {block_id} (index {i}): {len(in_arcs)} in-arcs, {len(out_arcs)} out-arcs")
        print(f"  In-arcs: {in_arcs[:3]}...")  # Show first 3
        print(f"  Out-arcs: {out_arcs[:3]}...")  # Show first 3
    
    # Create a small master problem to test flow indexing
    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    k = 2  # 2 districts
    arc_list = geodata.get_arc_list()
    
    print(f"\nCreating test master problem with {N} blocks, {k} districts")
    print(f"Arc list contains {len(arc_list)} arcs")
    
    master = gp.Model("test_flow_indexing")
    master.setParam('OutputFlag', 1)  # Enable output for debugging
    
    # Create assignment variables
    z = master.addVars(N, N, vtype=GRB.BINARY, name="z")
    
    # Basic assignment constraints
    for j in range(N):
        master.addConstr(gp.quicksum(z[i, j] for i in range(N)) == 1)
    master.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k)
    for i in range(N):
        for j in range(N):
            master.addConstr(z[j, i] <= z[i, i])
    
    print(f"Added basic assignment constraints")
    
    # Create flow variables - this is where the issue likely is
    flow = {}
    flow_count = 0
    for root_id in block_ids:
        for arc in arc_list:
            flow[root_id, arc] = master.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                             name=f"flow_{root_id}_{arc[0]}_{arc[1]}")
            flow_count += 1
    
    print(f"Created {flow_count} flow variables")
    master.update()
    
    # Now add flow constraints - examine the indexing carefully
    constraint_count = 0
    for i, root_id in enumerate(block_ids):
        print(f"\nProcessing root {root_id} (index {i})")
        
        for v, block_v in enumerate(block_ids):
            in_arcs = geodata.get_in_arcs(block_v)
            out_arcs = geodata.get_out_arcs(block_v)
            
            if block_v == root_id:
                # Root constraint: inflow = 0
                if in_arcs:
                    print(f"  Root constraint for {block_v}: {len(in_arcs)} in-arcs")
                    # Check if flow variables exist for these arcs
                    for arc in in_arcs:
                        if (root_id, arc) not in flow:
                            print(f"    ERROR: Flow variable missing for root={root_id}, arc={arc}")
                        else:
                            print(f"    OK: Flow variable exists for root={root_id}, arc={arc}")
                    
                    master.addConstr(gp.quicksum(flow[root_id, arc] for arc in in_arcs) == 0)
                    constraint_count += 1
            else:
                # Non-root constraint: inflow - outflow = z[v, i]
                print(f"  Non-root constraint for {block_v} (index {v}): {len(in_arcs)} in, {len(out_arcs)} out")
                
                # Check flow variable existence
                missing_flow_vars = []
                for arc in in_arcs:
                    if (root_id, arc) not in flow:
                        missing_flow_vars.append(f"in-arc {arc}")
                for arc in out_arcs:
                    if (root_id, arc) not in flow:
                        missing_flow_vars.append(f"out-arc {arc}")
                
                if missing_flow_vars:
                    print(f"    ERROR: Missing flow variables: {missing_flow_vars[:3]}...")
                else:
                    print(f"    OK: All flow variables exist")
                
                # Add the constraint
                inflow = gp.quicksum(flow[root_id, arc] for arc in in_arcs) if in_arcs else 0
                outflow = gp.quicksum(flow[root_id, arc] for arc in out_arcs) if out_arcs else 0
                master.addConstr(inflow - outflow == z[v, i])  # This is the key constraint
                constraint_count += 1
                
                # Capacity constraint
                if in_arcs:
                    master.addConstr(gp.quicksum(flow[root_id, arc] for arc in in_arcs) <= (N-1) * z[v, i])
                    constraint_count += 1
    
    print(f"\nAdded {constraint_count} flow constraints")
    
    # Try to solve
    master.setObjective(0, GRB.MINIMIZE)  # Dummy objective
    print("\nAttempting to solve...")
    master.optimize()
    
    print(f"Optimization status: {master.status}")
    if master.status == GRB.INFEASIBLE:
        print("Model is infeasible - computing IIS...")
        master.computeIIS()
        master.write("test_flow_indexing_infeasible.ilp")
        print("IIS written to test_flow_indexing_infeasible.ilp")
    elif master.status == GRB.OPTIMAL:
        print("Model solved successfully!")
        
        # Check solution
        print("\nSolution analysis:")
        roots = [i for i in range(N) if round(z[i, i].X) == 1]
        print(f"Selected roots: {[block_ids[i] for i in roots]}")
        
        for root_idx in roots:
            assigned = [j for j in range(N) if round(z[root_idx, j].X) == 1]
            print(f"District {block_ids[root_idx]}: {len(assigned)} blocks assigned")

if __name__ == "__main__":
    test_flow_indexing() 