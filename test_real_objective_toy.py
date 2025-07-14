#!/usr/bin/env python3
import numpy as np
from test_toy_model_25blocks import ToyGeoData
from lib.algorithm import Partition
import matplotlib.pyplot as plt

# 1. Create toy geodata (5x5 grid)
toy_geo = ToyGeoData(n_blocks=25, grid_size=5, seed=42)

# 2. Generate a random probability distribution (uniform)
def generate_random_probabilities(block_ids, seed=42):
    np.random.seed(seed)
    probs = np.random.uniform(0.01, 0.1, len(block_ids))
    probs = probs / probs.sum()
    return {bid: p for bid, p in zip(block_ids, probs)}

prob_dict = generate_random_probabilities(toy_geo.short_geoid_list, seed=42)

# 3. Compute K for all blocks (use depot at (0,0) for toy)
toy_geo.compute_K_for_all_blocks(depot_lat=0, depot_lon=0)

# 4. Create Partition instance
num_districts = 3
partition = Partition(toy_geo, num_districts, prob_dict, epsilon=0.1)

# Utility: expand assignment matrix for evaluate_real_objective
def expand_assignment_matrix(block_assignment, block_centers, all_blocks):
    N = len(all_blocks)
    num_districts = len(block_centers)
    expanded = np.zeros((N, N))
    center_idx_map = {center: all_blocks.index(center) for center in block_centers}
    for j in range(N):
        for d in range(num_districts):
            expanded[j, center_idx_map[block_centers[d]]] = block_assignment[j, d]
    return expanded

# 5. Run random_search and local_search with real objective
Lambda = 1.0
wr = 1.0
wv = 10.0
beta = 0.7120

print("Running random search...")
best_centers, best_assignment, best_obj_val = partition.random_search(
    max_iters=20, prob_dict=prob_dict, epsilon=0.1, Lambda=Lambda, wr=wr, wv=wv, beta=beta
)
# Expand assignment for evaluation/visualization
expanded_assignment = expand_assignment_matrix(best_assignment, best_centers, toy_geo.short_geoid_list)

print("Running local search...")
best_centers, best_assignment, best_obj_val = partition.local_search(
    best_centers, best_obj_val, prob_dict=prob_dict, epsilon=0.1, Lambda=Lambda, wr=wr, wv=wv, beta=beta
)
expanded_assignment = expand_assignment_matrix(best_assignment, best_centers, toy_geo.short_geoid_list)

print(f"Final best objective value: {best_obj_val}")

# 6. Visualize the result
toy_geo.plot_topology(expanded_assignment) 