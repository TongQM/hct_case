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
    """Simplified GeoData class for toy model testing"""
    
    def __init__(self, n_blocks=25, grid_size=5, seed=42):
        """
        Create a toy geographical dataset with n_blocks arranged in a grid
        
        Parameters:
        - n_blocks: Number of blocks (should be perfect square for grid)
        - grid_size: Size of the grid (grid_size x grid_size = n_blocks)
        - seed: Random seed for reproducibility
        """
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
        
        # Create a mock gdf (GeoDataFrame) for compatibility with Partition class
        self._create_mock_gdf()
        
        print(f"Created toy model with {n_blocks} blocks in {grid_size}x{grid_size} grid")
        print(f"Graph has {len(self.G.edges())} edges, {len(self.arc_list)} directed arcs")
    
    def _create_grid_topology(self):
        """Create a grid topology for the blocks"""
        self.grid_positions = {}
        
        # Arrange blocks in a grid
        for i, block_id in enumerate(self.short_geoid_list):
            row = i // self.grid_size
            col = i % self.grid_size
            self.grid_positions[block_id] = (row, col)
    
    def _build_connectivity(self):
        """Build connectivity graph based on grid adjacency"""
        self.G = nx.Graph()
        
        # Add all nodes
        for block_id in self.short_geoid_list:
            self.G.add_node(block_id)
        
        # Add edges for adjacent blocks (4-connectivity: up, down, left, right)
        for block_id in self.short_geoid_list:
            row, col = self.grid_positions[block_id]
            
            # Check all 4 directions
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                new_row, new_col = row + dr, col + dc
                
                # Check if neighbor is within bounds
                if 0 <= new_row < self.grid_size and 0 <= new_col < self.grid_size:
                    neighbor_idx = new_row * self.grid_size + new_col
                    if neighbor_idx < len(self.short_geoid_list):
                        neighbor_id = self.short_geoid_list[neighbor_idx]
                        self.G.add_edge(block_id, neighbor_id)
        
        # Add some random long-distance connections to make it more interesting
        n_random_edges = max(1, len(self.short_geoid_list) // 10)
        for _ in range(n_random_edges):
            block1 = random.choice(self.short_geoid_list)
            block2 = random.choice(self.short_geoid_list)
            if block1 != block2:
                self.G.add_edge(block1, block2)
        
        print(f"Grid connectivity: {len(self.G.edges())} edges")
    
    def _build_arc_structures(self):
        """Build directed arc list and precompute in/out arc dictionaries"""
        # Build directed arc list from undirected graph
        self.arc_list = []
        for u, v in self.G.edges():
            self.arc_list.append((u, v))
            self.arc_list.append((v, u))
        self.arc_list = list(set(self.arc_list))  # Remove duplicates if any
        
        # Precompute out_arcs_dict and in_arcs_dict for all nodes
        self.out_arcs_dict = {
            node: [(node, neighbor) for neighbor in self.G.neighbors(node) if (node, neighbor) in self.arc_list] 
            for node in self.short_geoid_list
        }
        self.in_arcs_dict = {
            node: [(neighbor, node) for neighbor in self.G.neighbors(node) if (neighbor, node) in self.arc_list] 
            for node in self.short_geoid_list
        }
        
        logging.info(f"Built arc structures: {len(self.arc_list)} directed arcs for {len(self.short_geoid_list)} nodes")
    
    def get_arc_list(self):
        """Get the directed arc list"""
        return self.arc_list
    
    def get_in_arcs(self, node):
        """Get incoming arcs for a specific node"""
        return self.in_arcs_dict.get(node, [])
    
    def get_out_arcs(self, node):
        """Get outgoing arcs for a specific node"""
        return self.out_arcs_dict.get(node, [])
    
    def get_area(self, block_id):
        """Get area for a block"""
        return self.areas[block_id]
    
    def get_dist(self, block1, block2):
        """Get distance between two blocks (Manhattan distance in grid)"""
        if block1 == block2:
            return 0.0
        
        pos1 = self.grid_positions[block1]
        pos2 = self.grid_positions[block2]
        
        # Manhattan distance in grid units
        manhattan_dist = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
        
        # Convert to reasonable distance scale (multiply by average block size)
        avg_block_size = 1.0  # 1 km per grid unit
        return manhattan_dist * avg_block_size
    
    def plot_topology(self, partition_result=None, save_path=None):
        """Plot the grid topology and optionally the partition result"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot grid
        pos = {block_id: self.grid_positions[block_id] for block_id in self.short_geoid_list}
        
        if partition_result is not None:
            # Color nodes by district
            z_sol = partition_result
            N = len(self.short_geoid_list)
            
            # Find district assignments and create unique district mapping
            district_assignments = {}
            unique_districts = set()
            
            for j, block_id in enumerate(self.short_geoid_list):
                assigned_district = -1
                for i in range(N):
                    if round(z_sol[j, i]) == 1:
                        assigned_district = i
                        break
                district_assignments[block_id] = assigned_district
                unique_districts.add(assigned_district)
            
            # Map unique district indices to contiguous color indices
            unique_districts = sorted(list(unique_districts))
            district_to_color_idx = {dist: idx for idx, dist in enumerate(unique_districts)}
            n_districts = len(unique_districts)
            
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors
            
            if n_districts <= 10:
                base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
                color_map = {dist: base_colors[idx] for dist, idx in district_to_color_idx.items()}
            else:
                cmap = cm.get_cmap('tab20' if n_districts <= 20 else 'hsv')
                color_map = {dist: cmap(idx / max(1, n_districts-1)) for dist, idx in district_to_color_idx.items()}
            
            # Assign colors to nodes using the mapped color index
            colors = [color_map[district_assignments[block_id]] for block_id in self.short_geoid_list]
            
            nx.draw(self.G, pos, node_color=colors, with_labels=True, node_size=500, font_size=8, ax=ax)
            
            # Add legend showing district assignments
            legend_elements = []
            for dist in unique_districts:
                count = sum(1 for d in district_assignments.values() if d == dist)
                block_name = self.short_geoid_list[dist]
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                                markerfacecolor=color_map[dist], markersize=10, 
                                                label=f'District {block_name} ({count} blocks)'))
            
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1))
        else:
            nx.draw(self.G, pos, with_labels=True, node_size=500, font_size=8, ax=ax)
        
        ax.set_title(f"Toy Model: {self.n_blocks} Blocks in {self.grid_size}x{self.grid_size} Grid")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Topology plot saved to {save_path}")
        
        plt.show()

    def _create_mock_gdf(self):
        """Create a mock GeoDataFrame for compatibility"""
        # Create simple point geometries for each block
        geometries = []
        for block_id in self.short_geoid_list:
            row, col = self.grid_positions[block_id]
            # Create a simple point geometry
            point = Point(col, row)
            geometries.append(point)
        
        # Create mock GeoDataFrame
        self.gdf = pd.DataFrame({
            'geometry': geometries,
            'area': [self.areas[block_id] for block_id in self.short_geoid_list]
        })
        self.gdf.index = self.short_geoid_list

    def compute_K_for_all_blocks(self, depot_lat=0, depot_lon=0):
        """
        Compute K_i for each block as the roundtrip Euclidean distance from the depot (depot_lat, depot_lon)
        to the block's grid position. Store in self.K_dict and as a 'K' column in self.gdf if available.
        """
        K_dict = {}
        for block_id in self.short_geoid_list:
            row, col = self.grid_positions[block_id]
            # Treat (row, col) as (y, x) in grid; depot is at (depot_lat, depot_lon)
            dist = ((row - depot_lat) ** 2 + (col - depot_lon) ** 2) ** 0.5
            K_dict[block_id] = 2 * dist  # roundtrip
        self.K_dict = K_dict
        # Optionally store in gdf if it exists
        if hasattr(self, 'gdf') and hasattr(self.gdf, '__setitem__'):
            self.gdf['K'] = [K_dict[bid] for bid in self.short_geoid_list]
        return K_dict

    def get_K(self, block):
        return self.K_dict.get(block, 0.0)

def generate_random_probabilities(block_ids, seed=42, distribution='uniform'):
    """Generate random probability distributions for blocks"""
    np.random.seed(seed)
    
    if distribution == 'uniform':
        # Uniform distribution
        probs = np.random.uniform(0.01, 0.1, len(block_ids))
    elif distribution == 'exponential':
        # Exponential distribution (some blocks much higher demand)
        probs = np.random.exponential(0.03, len(block_ids))
    elif distribution == 'clustered':
        # Clustered distribution (high demand in center, low on edges)
        probs = []
        for i, block_id in enumerate(block_ids):
            # Distance from center
            center = len(block_ids) // 2
            distance = abs(i - center)
            prob = 0.08 * np.exp(-distance / 5.0) + np.random.uniform(0.01, 0.02)
            probs.append(prob)
        probs = np.array(probs)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Normalize to sum to 1
    probs = probs / np.sum(probs)
    
    return {block_ids[i]: probs[i] for i in range(len(block_ids))}

def test_toy_model_25blocks():
    """Test the partition algorithm on a 25-block toy model"""
    
    print("=" * 60)
    print("TESTING TOY MODEL WITH 25 BLOCKS")
    print("=" * 60)
    
    # Create toy geographical data
    toy_geo = ToyGeoData(n_blocks=25, grid_size=5, seed=42)
    
    # Generate random probabilities
    prob_distributions = {
        'uniform': generate_random_probabilities(toy_geo.short_geoid_list, seed=42, distribution='uniform'),
        # 'exponential': generate_random_probabilities(toy_geo.short_geoid_list, seed=42, distribution='exponential'),
        # 'clustered': generate_random_probabilities(toy_geo.short_geoid_list, seed=42, distribution='clustered')
    }
    
    # Test different configurations
    test_configs = [
        {'name': 'small_uniform', 'districts': 3, 'prob_type': 'uniform', 'epsilon': 0.1},
        # {'name': 'small_exponential', 'districts': 3, 'prob_type': 'exponential', 'epsilon': 0.1},
        # {'name': 'medium_clustered', 'districts': 5, 'prob_type': 'clustered', 'epsilon': 0.05},
    ]
    
    results = {}
    
    for config in test_configs:
        print(f"\n--- Testing Configuration: {config['name']} ---")
        print(f"Districts: {config['districts']}, Probability: {config['prob_type']}, Epsilon: {config['epsilon']}")
        
        prob_dict = prob_distributions[config['prob_type']]
        
        # Print probability statistics
        probs = list(prob_dict.values())
        print(f"Probability stats - Min: {min(probs):.4f}, Max: {max(probs):.4f}, Mean: {np.mean(probs):.4f}, Std: {np.std(probs):.4f}")
        
        try:
            # Create partition object
            partition = Partition(toy_geo, num_districts=config['districts'], prob_dict=prob_dict, epsilon=config['epsilon'])
            
            # Run LBBD
            print("Running Benders decomposition...")
            best_partition, best_cost, history = partition.benders_decomposition(
                max_iterations=100, tolerance=1e-3, max_cuts=100, verbose=True
            )
            
            # Analyze results
            print(f"\n✅ RESULTS for {config['name']}:")
            print(f"   Best cost: {best_cost:.4f}")
            print(f"   Iterations: {len(history)}")
            print(f"   Final gap: {history[-1]['gap']:.4f}" if history else "N/A")
            
            # Analyze district balance
            N = len(toy_geo.short_geoid_list)
            roots = [i for i in range(N) if round(best_partition[i, i]) == 1]
            district_sizes = []
            district_costs = []
            
            for root_idx in roots:
                assigned = [j for j in range(N) if round(best_partition[j, root_idx]) == 1]
                district_sizes.append(len(assigned))
                
                # Calculate district cost
                assigned_blocks = [toy_geo.short_geoid_list[j] for j in assigned]
                cost, _, _, _, _ = partition._SDP_benders(assigned_blocks, toy_geo.short_geoid_list[root_idx], prob_dict, config['epsilon'])
                district_costs.append(cost)
                
                print(f"   District {toy_geo.short_geoid_list[root_idx]}: {len(assigned)} blocks, cost: {cost:.4f}")
            
            balance_score = np.std(district_sizes)
            print(f"   District sizes: {district_sizes} (std: {balance_score:.2f})")
            print(f"   District costs: {[f'{c:.4f}' for c in district_costs]}")
            
            results[config['name']] = {
                'best_cost': best_cost,
                'iterations': len(history),
                'district_sizes': district_sizes,
                'district_costs': district_costs,
                'balance_score': balance_score,
                'partition': best_partition,
                'history': history
            }
            
            # Plot result
            save_path = f"figures/toy_model_{config['name']}.png"
            os.makedirs("figures", exist_ok=True)
            toy_geo.plot_topology(best_partition, save_path)
            
        except Exception as e:
            print(f"❌ ERROR in {config['name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            results[config['name']] = {'error': str(e)}
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL TESTS")
    print("=" * 60)
    
    for config_name, result in results.items():
        if 'error' in result:
            print(f"{config_name}: ❌ FAILED - {result['error']}")
        else:
            print(f"{config_name}: ✅ SUCCESS")
            print(f"  Cost: {result['best_cost']:.4f}, Iterations: {result['iterations']}")
            print(f"  Balance: {result['balance_score']:.2f}, Sizes: {result['district_sizes']}")
    
    return results

def test_connectivity_only():
    """Test just the connectivity constraints without SDP complexity"""
    print("\n--- Testing Connectivity-Only Model ---")
    
    toy_geo = ToyGeoData(n_blocks=9, grid_size=3, seed=42)  # Smaller for debugging
    
    import gurobipy as gp
    from gurobipy import GRB
    
    block_ids = toy_geo.short_geoid_list
    N = len(block_ids)
    k = 2  # districts
    
    # Get arc structures
    arc_list = toy_geo.get_arc_list()
    print(f"Using {len(arc_list)} directed arcs for {N} blocks")
    
    model = gp.Model("connectivity_test")
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
        model.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in toy_geo.get_in_arcs(root_id)) == 0, name=f"no_return_flow_{root_id}")
        for j, block_id in enumerate(block_ids):
            if block_id != root_id:
                model.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in toy_geo.get_in_arcs(block_id)) - gp.quicksum(f[root_id, arc[0], arc[1]] for arc in toy_geo.get_out_arcs(block_id)) == z[j, i], name=f"flow_assign_{block_id}_{root_id}")
                model.addConstr(gp.quicksum(f[root_id, arc[0], arc[1]] for arc in toy_geo.get_in_arcs(block_id)) <= (N - 1) * z[j, i], name=f"flow_restrict_{block_id}_{root_id}")
    
    # Simple objective: minimize sum of district sizes squared (encourage balance)
    district_sizes = [gp.quicksum(z[j, i] for j in range(N)) for i in range(N)]
    model.setObjective(gp.quicksum(size * size for size in district_sizes), GRB.MINIMIZE)
    
    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("✅ Connectivity test solved successfully")
        
        # Check solution
        roots = [i for i in range(N) if round(z[i, i].X) == 1]
        print(f"Found {len(roots)} districts with roots: {[block_ids[i] for i in roots]}")
        
        district_sizes = []
        for root_idx in roots:
            assigned = [j for j in range(N) if round(z[j, root_idx].X) == 1]
            district_sizes.append(len(assigned))
            assigned_names = [block_ids[j] for j in assigned]
            print(f"District {block_ids[root_idx]}: {len(assigned)} blocks = {assigned_names}")
        
        print(f"District sizes: {district_sizes}")
        return True
    else:
        print(f"❌ Connectivity test failed. Status: {model.status}")
        return False

if __name__ == "__main__":
    # First test basic connectivity
    print("Testing basic connectivity constraints...")
    if test_connectivity_only():
        print("\n" + "="*60)
        print("Basic connectivity test passed. Running full toy model tests...")
        test_toy_model_25blocks()
    else:
        print("Basic connectivity test failed. Please check the implementation.") 