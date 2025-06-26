#!/usr/bin/env python3

import gurobipy as gp
from gurobipy import GRB

def test_simple_flow():
    """Simple test to verify flow variable bounds"""
    
    model = gp.Model("test_bounds")
    model.setParam('OutputFlag', 1)
    
    # Create some flow variables with explicit bounds
    flow_vars = {}
    for i in range(3):
        for j in range(3):
            flow_vars[i, j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"flow_{i}_{j}")
    
    print(f"Created {len(flow_vars)} flow variables")
    
    # Add some simple constraints
    model.addConstr(flow_vars[0, 1] + flow_vars[0, 2] == 1)
    model.addConstr(flow_vars[1, 0] - flow_vars[1, 2] == 0)
    model.addConstr(flow_vars[2, 0] - flow_vars[2, 1] == -1)
    
    model.update()
    
    # Check variable bounds before optimization
    print("\nVariable bounds:")
    for key, var in flow_vars.items():
        print(f"  {var.VarName}: lb={var.LB}, ub={var.UB}")
    
    # Set dummy objective
    model.setObjective(0, GRB.MINIMIZE)
    
    # Try to solve
    model.optimize()
    
    print(f"\nOptimization status: {model.status}")
    if model.status == GRB.INFEASIBLE:
        print("Model is infeasible - computing IIS...")
        model.computeIIS()
        model.write("test_simple_flow_infeasible.ilp")
        print("IIS written to test_simple_flow_infeasible.ilp")
    elif model.status == GRB.OPTIMAL:
        print("Model solved successfully!")
        for key, var in flow_vars.items():
            print(f"  {var.VarName} = {var.X}")

if __name__ == "__main__":
    test_simple_flow() 