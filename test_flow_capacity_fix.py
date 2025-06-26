#!/usr/bin/env python3

import sys
import logging
import geopandas as gpd
import gurobipy as gp
from gurobipy import GRB
import networkx as nx

# Add lib directory to path
sys.path.append('lib')
from data import GeoData

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_flow_capacity_fix():
    """Test flow-based contiguity with capacity fix on a medium connected subset"""
    
    print("Loading GeoData and finding connected subset...")
    
    # Load full shapefile
    shapefile_path = "2023_shape_files/census_shape_block_group/tl_2023_42_bg.shp"
    gdf = gpd.read_file(shapefile_path)
    gdf['short_GEOID'] = gdf['GEOID'].str[-6:]
    
    # Use the first 20 GEOIDs from the actual available list (from data_process.ipynb)
    available_geoids = gdf['short_GEOID'].tolist()
    test_geoids = available_geoids[:20]  # First 20 blocks
    print(f"Testing with first {len(test_geoids)} blocks: {test_geoids[:5]}...")
    
    # Initialize GeoData
    geodata = GeoData(shapefile_path, test_geoids)
    
    # Find largest connected component
    G = nx.Graph()
    for arc in geodata.get_arc_list():
        G.add_edge(arc[0], arc[1])
    
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest_component = max(components, key=len)
        test_geoids = list(largest_component)
        print(f"Using largest connected component: {len(test_geoids)} blocks")
        
        # Reinitialize with connected subset
        geodata = GeoData(shapefile_path, test_geoids)
    else:
        print(f"All {len(test_geoids)} blocks are connected")
    
    print(f"Graph has {len(geodata.get_arc_list())} directed arcs")
    
    # Create flow-based contiguity model
    N = len(test_geoids)
    k = 2  # 2 districts
    
    print(f"\nCreating flow-based contiguity model: {N} blocks, {k} districts")
    
    model = gp.Model("flow_contiguity_test")
    model.setParam('OutputFlag', 1)  # Show Gurobi output
    
    # Assignment variables: z[i,j] = 1 if block j assigned to root i
    z = model.addVars(N, N, vtype=GRB.BINARY, name="z")
    
    # Basic assignment constraints
    for j in range(N):
        model.addConstr(gp.quicksum(z[i, j] for i in range(N)) == 1, name=f"assign_{j}")
    
    # Exactly k districts
    model.addConstr(gp.quicksum(z[i, i] for i in range(N)) == k, name="num_districts")
    
    # Validity constraints
    for i in range(N):
        for j in range(N):
            model.addConstr(z[i, j] <= z[i, i], name=f"valid_{i}_{j}")
    
    print(f"Added basic constraints: {N + 1 + N*N} constraints")
    
    # Flow variables and constraints
    arc_list = geodata.get_arc_list()
    flow = {}
    for root_id in test_geoids:
        for arc in arc_list:
            flow[root_id, arc] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, 
                                            name=f"flow_{root_id}_{arc[0]}_{arc[1]}")
    
    print(f"Created {len(flow)} flow variables")
    
    # Flow conservation constraints
    flow_constraints = 0
    for i, root_id in enumerate(test_geoids):
        for v, block_v in enumerate(test_geoids):
            in_arcs = geodata.get_in_arcs(block_v)
            out_arcs = geodata.get_out_arcs(block_v)
            
            if block_v == root_id:
                # Root has no inflow
                if in_arcs:
                    model.addConstr(gp.quicksum(flow[root_id, arc] for arc in in_arcs) == 0,
                                  name=f"root_inflow_{root_id}")
                    flow_constraints += 1
            else:
                # Non-root: inflow - outflow = z[i, v]
                inflow = gp.quicksum(flow[root_id, arc] for arc in in_arcs) if in_arcs else 0
                outflow = gp.quicksum(flow[root_id, arc] for arc in out_arcs) if out_arcs else 0
                model.addConstr(inflow - outflow == z[i, v], 
                              name=f"flow_balance_{root_id}_{block_v}")
                flow_constraints += 1
                
                # Capacity constraint: inflow <= (N-1) * z[i, v]
                if in_arcs:
                    model.addConstr(gp.quicksum(flow[root_id, arc] for arc in in_arcs) <= (N-1) * z[i, v],
                                  name=f"capacity_in_{root_id}_{block_v}")
                    flow_constraints += 1
    
    # Arc capacity constraints (THE FIX)
    capacity_constraints = 0
    for root_id in test_geoids:
        i = test_geoids.index(root_id)
        for (u, v) in arc_list:
            if u in test_geoids:
                u_idx = test_geoids.index(u)
                # FIXED: z[i, u_idx] with capacity 1 (not N-1)
                model.addConstr(flow[root_id, (u, v)] <= z[i, u_idx],
                              name=f"arc_capacity_{root_id}_{u}_{v}")
                capacity_constraints += 1
    
    print(f"Added flow constraints: {flow_constraints} flow + {capacity_constraints} capacity = {flow_constraints + capacity_constraints} total")
    print(f"Model has {model.NumVars} variables and {model.NumConstrs} constraints")
    
    # Set dummy objective
    model.setObjective(0, GRB.MINIMIZE)
    
    print("\nSolving...")
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("✓ Model solved successfully!")
        
        # Analyze solution
        roots = [i for i in range(N) if round(z[i, i].X) == 1]
        print(f"Selected roots: {[test_geoids[i] for i in roots]}")
        
        for root in roots:
            assigned = [j for j in range(N) if round(z[root, j].X) == 1]
            print(f"District {test_geoids[root]}: {len(assigned)} blocks")
        
        return True
        
    elif model.status == GRB.INFEASIBLE:
        print("✗ Model is infeasible")
        model.computeIIS()
        model.write("flow_capacity_test_infeasible.ilp")
        print("IIS written to flow_capacity_test_infeasible.ilp")
        return False
        
    else:
        print(f"✗ Model failed with status {model.status}")
        return False

if __name__ == "__main__":
    success = test_flow_capacity_fix()
    if success:
        print("\n✓ Flow capacity fix test PASSED")
    else:
        print("\n✗ Flow capacity fix test FAILED")
        sys.exit(1) 