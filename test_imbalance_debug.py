#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lib.data import GeoData
from lib.algorithm import Partition
import numpy as np
import logging
from lbbd_config import SMALL_BG_GEOID_LIST, DATA_PATHS

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_imbalance_issue():
    """Analyze why the LBBD creates imbalanced partitions"""
    
    print("=== Analyzing LBBD Imbalance Issue ===")
    
    # Use the same configuration as tiny_test
    geodata = GeoData(
        filepath=DATA_PATHS['shapefile'],
        geoid_list=SMALL_BG_GEOID_LIST,
        level='block_group'
    )
    
    print(f"Loaded {len(geodata.short_geoid_list)} blocks: {geodata.short_geoid_list}")
    
    # Create uniform probability distribution
    prob_dict = {block: 1.0/len(geodata.short_geoid_list) for block in geodata.short_geoid_list}
    
    # Initialize partition with 2 districts
    partition = Partition(geodata, num_districts=2, prob_dict=prob_dict, epsilon=0.1)
    
    print("\n=== Running Full LBBD ===")
    best_partition, best_cost, history = partition.benders_decomposition(
        max_iterations=3, tolerance=1e-2, max_cuts=10, verbose=True
    )
    
    print(f"\nFinal cost: {best_cost:.4f}")
    
    # Analyze the final partition
    print("\n=== Analyzing Final Partition ===")
    analyze_partition_balance(best_partition, geodata.short_geoid_list)
    
    # Test what happens with different objectives
    print("\n=== Testing Alternative Objectives ===")
    test_balanced_objective(geodata, prob_dict)

def analyze_partition_balance(z_sol, block_ids):
    """Analyze the balance of a partition solution"""
    N = len(block_ids)
    
    # Find roots
    roots = [i for i in range(N) if round(z_sol[i, i]) == 1]
    print(f"Number of districts: {len(roots)}")
    
    district_sizes = []
    for root_idx in roots:
        assigned = []
        for j in range(N):
            if round(z_sol[j, root_idx]) == 1:
                assigned.append(block_ids[j])
        district_sizes.append(len(assigned))
        print(f"District {block_ids[root_idx]}: {len(assigned)} blocks = {assigned}")
    
    print(f"District sizes: {district_sizes}")
    if len(district_sizes) > 1:
        size_variance = np.var(district_sizes)
        size_range = max(district_sizes) - min(district_sizes)
        print(f"Size variance: {size_variance:.2f}, Size range: {size_range}")
        print(f"Balanced: {size_range <= 1}")
    
    return district_sizes

def test_balanced_objective(geodata, prob_dict):
    """Test what happens with a balanced objective instead of worst-case"""
    
    print("Testing master problem with balanced objective...")
    
    block_ids = geodata.short_geoid_list
    N = len(block_ids)
    k = 2
    
    import gurobipy as gp
    from gurobipy import GRB
    
    # Get arc structures
    arc_list = geodata.get_arc_list()
    
    model = gp.Model("balanced_objective")
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
        model.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(root_id)) == 0, name=f"no_return_flow_{root_id}")
        for j, block_id in enumerate(block_ids):
            if block_id != root_id:
                model.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(block_id)) - gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_out_arcs(block_id)) == z[j, i], name=f"flow_assign_{block_id}_{root_id}")
                model.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in geodata.get_in_arcs(block_id)) <= (N - 1) * z[j, i], name=f"flow_restrict_{block_id}_{root_id}")
    
    # BALANCED OBJECTIVE: minimize variance of district sizes
    district_sizes = [gp.quicksum(z[j, i] for j in range(N)) for i in range(N)]
    avg_size = N / k
    model.setObjective(gp.quicksum((size - avg_size) * (size - avg_size) for size in district_sizes), GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("✓ Balanced objective model solved successfully")
        
        # Get solution
        z_sol = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                z_sol[j, i] = z[j, i].X
        
        print("Partition with balanced objective:")
        district_sizes = analyze_partition_balance(z_sol, block_ids)
        
        # Now evaluate this partition with the actual SDP costs
        print("\nEvaluating SDP costs for balanced partition:")
        partition = Partition(geodata, num_districts=k, prob_dict=prob_dict, epsilon=0.1)
        
        roots = [i for i in range(N) if round(z_sol[i, i]) == 1]
        district_costs = []
        
        for root in roots:
            assigned_blocks = [j for j in range(N) if round(z_sol[j, root]) == 1]
            assigned_block_ids = [block_ids[j] for j in assigned_blocks]
            
            cost, x_star, subgrad, T_star, alpha_i = partition._SDP_benders(
                assigned_block_ids, block_ids[root], prob_dict, partition.epsilon)
            district_costs.append(cost)
            print(f"District {block_ids[root]} ({len(assigned_blocks)} blocks): cost = {cost:.4f}")
        
        worst_cost = max(district_costs)
        print(f"Worst-case cost with balanced partition: {worst_cost:.4f}")
        print(f"District costs: {[f'{c:.4f}' for c in district_costs]}")
        
    else:
        print(f"✗ Balanced objective model failed. Status: {model.status}")

def main():
    """Main analysis function"""
    print("🔍 HYPOTHESIS: LBBD creates imbalanced partitions because it optimizes worst-case cost")
    print("   This can favor large districts if they're 'efficient' in worst-case terms")
    print("   even if the partition is highly imbalanced.\n")
    
    analyze_imbalance_issue()
    
    print("\n" + "="*60)
    print("CONCLUSION:")
    print("- LBBD minimizes worst-case district cost")
    print("- This doesn't inherently enforce balance")
    print("- Large districts might be 'efficient' due to economies of scale")
    print("- Consider adding balance constraints or multi-objective optimization")
    print("="*60)

if __name__ == "__main__":
    main() 