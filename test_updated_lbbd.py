#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
import geopandas as gpd
from shapely.geometry import Point, Polygon
import pandas as pd
import matplotlib.pyplot as plt
import random
from lib.data import GeoData
from lib.algorithm import Partition
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ToyGeoData:
    """Simplified GeoData class for testing updated LBBD"""
    
    def __init__(self, n_blocks=9, grid_size=3, seed=42):
        """Create a toy geographical dataset with n_blocks arranged in a grid"""
        np.random.seed(seed)
        random.seed(seed)
        
        self.n_blocks = n_blocks
        self.grid_size = grid_size
        
        # Generate block IDs
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
        
        # Create grid topology
        self._create_grid_topology()
        
        # Generate random areas (between 0.5 and 2.0 sq km)
        self.areas = {block_id: np.random.uniform(0.5, 2.0) for block_id in self.short_geoid_list}
        
        # Build connectivity graph and arc structures
        self._build_connectivity()
        self._build_arc_structures()
        
        # Initialize K and F values (will be overridden by partition-dependent costs)
        self.K_values = {block_id: 0.0 for block_id in self.short_geoid_list}
        self.F_values = {block_id: 0.0 for block_id in self.short_geoid_list}
        
        # Add required attributes for compatibility
        self._create_mock_gdf()
        self.wr = 1.0  # Default weight parameter
        self.wv = 1.0  # Default weight parameter
    
    def _create_grid_topology(self):
        # Create centroids on a grid
        self.centroids = {}
        for i in range(self.n_blocks):
            row = i // self.grid_size
            col = i % self.grid_size
            x = col * 1.0  # 1 km spacing
            y = row * 1.0
            self.centroids[self.short_geoid_list[i]] = (x, y)
    
    def _build_connectivity(self):
        # Build graph with 4-connectivity (up, down, left, right)
        self.G = nx.Graph()
        self.G.add_nodes_from(self.short_geoid_list)
        
        for i in range(self.n_blocks):
            row = i // self.grid_size
            col = i % self.grid_size
            current_block = self.short_geoid_list[i]
            
            # Add edges to neighbors (4-connectivity)
            neighbors = []
            if row > 0:  # up
                neighbors.append((row-1) * self.grid_size + col)
            if row < self.grid_size - 1:  # down
                neighbors.append((row+1) * self.grid_size + col)
            if col > 0:  # left
                neighbors.append(row * self.grid_size + (col-1))
            if col < self.grid_size - 1:  # right
                neighbors.append(row * self.grid_size + (col+1))
            
            for neighbor_idx in neighbors:
                neighbor_block = self.short_geoid_list[neighbor_idx]
                self.G.add_edge(current_block, neighbor_block)
    
    def _build_arc_structures(self):
        # Build directed arcs from undirected edges
        self.arc_list = []
        self.in_arcs_dict = {block_id: [] for block_id in self.short_geoid_list}
        self.out_arcs_dict = {block_id: [] for block_id in self.short_geoid_list}
        
        for u, v in self.G.edges():
            # Forward arc
            arc_forward = (u, v)
            self.arc_list.append(arc_forward)
            self.out_arcs_dict[u].append(arc_forward)
            self.in_arcs_dict[v].append(arc_forward)
            
            # Backward arc
            arc_backward = (v, u)
            self.arc_list.append(arc_backward)
            self.out_arcs_dict[v].append(arc_backward)
            self.in_arcs_dict[u].append(arc_backward)
    
    def get_arc_list(self):
        return self.arc_list
    
    def get_in_arcs(self, block_id):
        return self.in_arcs_dict.get(block_id, [])
    
    def get_out_arcs(self, block_id):
        return self.out_arcs_dict.get(block_id, [])
    
    def get_dist(self, block1, block2):
        # Euclidean distance between centroids
        x1, y1 = self.centroids[block1]
        x2, y2 = self.centroids[block2]
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def get_area(self, block_id):
        return self.areas[block_id]
    
    def get_K(self, block_id):
        return self.K_values.get(block_id, 0.0)
    
    def get_F(self, block_id):
        return self.F_values.get(block_id, 0.0)
    
    def _create_mock_gdf(self):
        """Create a mock GeoDataFrame for compatibility"""
        # Create simple point geometries for each block
        geometries = []
        for block_id in self.short_geoid_list:
            x, y = self.centroids[block_id]
            geometries.append(Point(x, y))
        
        self.gdf = gpd.GeoDataFrame({
            'GEOID20': self.short_geoid_list,
            'geometry': geometries
        })
    
    def compute_K_for_all_blocks(self, depot_coords):
        """Compute K_i (roundtrip distance to depot) for all blocks"""
        K_dict = {}
        for block_id in self.short_geoid_list:
            x, y = self.centroids[block_id]
            dist = np.sqrt((x - depot_coords[0])**2 + (y - depot_coords[1])**2)
            K_dict[block_id] = 2.0 * dist  # roundtrip
        return K_dict


def test_updated_lbbd():
    """Test the updated LBBD with depot location decisions and partition-dependent costs"""
    
    print("=== TESTING UPDATED LBBD ===")
    print("Features: Depot location decisions, partition-dependent costs (K_i, F_i)")
    
    # Create toy model
    toy_geo = ToyGeoData(9, 3)  # 3x3 grid
    
    # Setup probability distribution  
    np.random.seed(42)
    prob_dict = {}
    for block_id in toy_geo.short_geoid_list:
        prob_dict[block_id] = np.random.uniform(0.05, 0.15)
    
    # Normalize probabilities
    total_prob = sum(prob_dict.values())
    for k in prob_dict:
        prob_dict[k] /= total_prob
    
    print(f"Blocks: {toy_geo.short_geoid_list}")
    print(f"Probability distribution: {[f'{prob_dict[bid]:.3f}' for bid in toy_geo.short_geoid_list]}")
    
    # Add required parameters for current LBBD
    toy_geo.beta = 0.7120  # BHH coefficient
    toy_geo.Lambda = 10.0  # Overall arrival rate
    
    # Create partition instance
    partition = Partition(toy_geo, num_districts=2, prob_dict=prob_dict, epsilon=0.1)
    
    # Define 2D ODD summary vectors (Omega_j for each block)
    Omega_dict = {}
    for i, block_id in enumerate(toy_geo.short_geoid_list):
        # Simulate 2D ODD features: [population_density, commercial_activity]
        row = i // 3
        col = i % 3
        if row == 1 and col == 1:  # center block
            Omega_dict[block_id] = [3.0, 2.5]  # high pop, high commercial
        elif (row == 1) or (col == 1):  # middle row/column
            Omega_dict[block_id] = [2.0, 1.5]  # medium pop, medium commercial
        else:  # corner blocks
            Omega_dict[block_id] = [1.0, 0.5]  # low pop, low commercial
    
    print(f"2D ODD vectors (Omega_j): {[f'[{Omega_dict[bid][0]:.1f},{Omega_dict[bid][1]:.1f}]' for bid in toy_geo.short_geoid_list]}")
    
    # Define linear ODD cost function J(omega) that is zero at zero
    def J_function(omega_vector):
        """Linear ODD cost function: J(ω) = w1*ω1 + w2*ω2, zero at zero"""
        if hasattr(omega_vector, '__len__') and len(omega_vector) >= 2:
            return 0.3 * omega_vector[0] + 0.2 * omega_vector[1]  # w1=0.3, w2=0.2
        else:
            return 0.0
    
    print(f"Linear ODD cost function: J(ω) = 0.3*ω1 + 0.2*ω2 (zero at zero)")
    
    # Run updated LBBD
    print("\n--- Running Updated LBBD ---")
    try:
        result = partition.benders_decomposition(
            max_iterations=3,  # Small number for testing
            tolerance=1e-3,
            max_cuts=20,
            verbose=True,
            Omega_dict=Omega_dict,
            J_function=J_function
        )
        
        print("\n=== LBBD RESULTS ===")
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Final gap: {result['final_gap']:.4f}")
        print(f"Best cost: {result['best_cost']:.4f}")
        print(f"Best depot: {result['best_depot']}")
        
        # Show district-wise costs
        print("\n=== DISTRICT COSTS ===")
        z_sol = result['best_partition']
        N = len(toy_geo.short_geoid_list)
        
        for i in range(N):
            if round(z_sol[i, i]) == 1:  # This is a root
                root_id = toy_geo.short_geoid_list[i]
                assigned_blocks = [j for j in range(N) if round(z_sol[j, i]) == 1]
                K_i = result['best_K'][i] if result['best_K'] is not None else 0.0
                F_i = result['best_F'][i] if result['best_F'] is not None else 0.0
                omega_i = result['best_omega'][i] if result['best_omega'] is not None else [0.0, 0.0]
                
                print(f"District {root_id}: {len(assigned_blocks)} blocks")
                print(f"   K_i = {K_i:.4f}, F_i = {F_i:.4f}")
                print(f"   ω_i = [{omega_i[0]:.2f}, {omega_i[1]:.2f}]")
        
        # Show convergence history
        if 'history' in result and result['history']:
            print("\n=== CONVERGENCE HISTORY ===")
            for h in result['history']:
                print(f"Iter {h['iteration']}: LB={h['lower_bound']:.4f}, UB={h['upper_bound']:.4f}, Gap={h['gap']:.4f}")
        
        # Show partition assignment
        print("\n=== PARTITION ASSIGNMENT ===")
        for i, block_id in enumerate(toy_geo.short_geoid_list):
            assigned_to = np.argmax(z_sol[i])
            root_block = toy_geo.short_geoid_list[assigned_to]
            print(f"Block {block_id} -> District rooted at {root_block}")
        
        print("\n✅ Updated LBBD test completed successfully!")
        return result
        
    except Exception as e:
        print(f"\n❌ Error in updated LBBD: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    test_updated_lbbd()

