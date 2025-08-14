#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import random
from lib.algorithm import Partition
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ServiceRegionGeoData:
    """
    GeoData class for 10×10 mile service region with 5×5 blocks (2×2 mile blocks each)
    """
    
    def __init__(self, grid_size=5, region_size=10.0, seed=42):
        """
        Create a service region with grid_size × grid_size blocks
        
        Parameters:
        - grid_size: Number of blocks per side (5 for 5×5 grid)
        - region_size: Total region size in miles (10 for 10×10 mile region)
        - seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        random.seed(seed)
        
        self.grid_size = grid_size
        self.region_size = region_size
        self.block_size = region_size / grid_size  # 2 miles per block
        self.n_blocks = grid_size * grid_size
        
        # Generate block IDs
        self.short_geoid_list = [f"BLK{i:02d}" for i in range(self.n_blocks)]
        
        # Create grid topology and features
        self._create_grid_topology()
        self._generate_block_features()
        
        # Build connectivity graph and arc structures
        self._build_connectivity()
        self._build_arc_structures()
        
        # Initialize cost values (will be determined by LBBD)
        self.K_values = {block_id: 0.0 for block_id in self.short_geoid_list}
        self.F_values = {block_id: 0.0 for block_id in self.short_geoid_list}
        
        # Add compatibility attributes for Partition class
        self.gdf = None  # Not used in this implementation
        self.wr = 1.0    # Rider cost weight
        self.wv = 10.0   # Vehicle cost weight
    
    def _create_grid_topology(self):
        """Create block centroids and boundaries"""
        self.centroids = {}
        self.boundaries = {}
        
        for i in range(self.n_blocks):
            row = i // self.grid_size
            col = i % self.grid_size
            block_id = self.short_geoid_list[i]
            
            # Centroid coordinates (miles)
            x = (col + 0.5) * self.block_size
            y = (row + 0.5) * self.block_size
            self.centroids[block_id] = (x, y)
            
            # Block boundaries for visualization
            x_min = col * self.block_size
            x_max = (col + 1) * self.block_size
            y_min = row * self.block_size
            y_max = (row + 1) * self.block_size
            self.boundaries[block_id] = (x_min, y_min, x_max, y_max)
    
    def _generate_block_features(self):
        """Generate 2D ODD features and other block characteristics"""
        self.areas = {}
        self.odd_features = {}  # 2D ODD feature space
        
        for i, block_id in enumerate(self.short_geoid_list):
            row = i // self.grid_size
            col = i % self.grid_size
            
            # Block area (slightly random around 4 sq miles)
            self.areas[block_id] = np.random.uniform(3.5, 4.5)
            
            # 2D ODD features: (population_density, commercial_activity)
            # Create spatial patterns
            center_distance = np.sqrt((row - 2)**2 + (col - 2)**2)  # Distance from center
            
            # Population density: higher in center, lower at edges
            pop_density = max(0.5, 3.0 - 0.3 * center_distance + np.random.normal(0, 0.2))
            
            # Commercial activity: clustered in certain areas
            if (row in [1, 3] and col in [1, 3]):  # Commercial clusters
                commercial = np.random.uniform(2.5, 4.0)
            elif (row == 2 and col == 2):  # City center
                commercial = np.random.uniform(3.5, 5.0)
            else:
                commercial = np.random.uniform(0.5, 2.0)
            
            self.odd_features[block_id] = np.array([pop_density, commercial])
    
    def _build_connectivity(self):
        """Build graph with 4-connectivity (up, down, left, right)"""
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
        """Build directed arcs from undirected edges"""
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
        """Manhattan distance between block centroids (miles)"""
        x1, y1 = self.centroids[block1]
        x2, y2 = self.centroids[block2]
        return abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance for city blocks
    
    def get_area(self, block_id):
        return self.areas[block_id]
    
    def get_K(self, block_id):
        return self.K_values.get(block_id, 0.0)
    
    def get_F(self, block_id):
        return self.F_values.get(block_id, 0.0)
    
    def get_odd_features(self, block_id):
        """Get 2D ODD features for a block"""
        return self.odd_features[block_id]


def create_odd_parameters(geo_data):
    """
    Create ODD parameters for the service region
    
    Returns:
    - Omega_dict: 2D ODD feature vectors for each block
    - J_function: ODD cost function that takes 2D vector input
    """
    # Each block has a 2D ODD feature vector [population_density, commercial_activity]
    Omega_dict = {}
    for block_id in geo_data.short_geoid_list:
        features = geo_data.get_odd_features(block_id)
        # Store as 2D vector: [population_density, commercial_activity]
        Omega_dict[block_id] = features  # numpy array [pop_density, commercial]
    
    # LINEAR ODD cost function: J(ω) where ω is 2D vector [ω1, ω2]
    # IMPORTANT: J(0) = 0 (zero at zero)
    def J_function(omega_vector):
        """
        LINEAR ODD cost function for 2D feature vector
        omega_vector: numpy array [population_density, commercial_activity]
        Returns: w1*ω1 + w2*ω2 (no base cost, zero at zero)
        """
        if isinstance(omega_vector, (int, float)):
            # Handle scalar input
            return 1.5 * omega_vector
        
        omega1, omega2 = omega_vector[0], omega_vector[1]
        w1, w2 = 1.2, 0.8  # Linear weights for population density and commercial activity
        return w1 * omega1 + w2 * omega2  # Linear, zero at zero
    
    return Omega_dict, J_function


def create_probability_distribution(geo_data):
    """Create realistic probability distribution based on block features"""
    prob_dict = {}
    
    for block_id in geo_data.short_geoid_list:
        features = geo_data.get_odd_features(block_id)
        area = geo_data.get_area(block_id)
        
        # Probability proportional to population density and area
        pop_density = features[0]
        prob_dict[block_id] = pop_density * area
    
    # Normalize to sum to 1
    total_prob = sum(prob_dict.values())
    for k in prob_dict:
        prob_dict[k] /= total_prob
    
    return prob_dict


def visualize_results(geo_data, result, prob_dict, Omega_dict):
    """
    Visualize the optimal partition, depot location, and dispatch intervals
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('LBBD Results: 10×10 Mile Service Region (5×5 Blocks)', fontsize=16, fontweight='bold')
    
    # Define colors for districts
    district_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Extract partition information
    z_sol = result['best_partition']
    depot_block = result['best_depot']
    K_costs = result['best_K']
    F_costs = result['best_F']
    omega_values = result['best_omega']
    
    # Identify districts and their roots
    districts = {}
    for i, block_id in enumerate(geo_data.short_geoid_list):
        assigned_to = np.argmax(z_sol[i])
        root_block = geo_data.short_geoid_list[assigned_to]
        if root_block not in districts:
            districts[root_block] = []
        districts[root_block].append(block_id)
    
    # Plot 1: Optimal Partition with Depot
    ax1 = axes[0, 0]
    ax1.set_title('Optimal Partition & Depot Location', fontsize=14, fontweight='bold')
    
    for district_idx, (root, blocks) in enumerate(districts.items()):
        color = district_colors[district_idx % len(district_colors)]
        
        for block_id in blocks:
            x_min, y_min, x_max, y_max = geo_data.boundaries[block_id]
            
            # Draw block boundary
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                   linewidth=2, edgecolor='black', facecolor=color, alpha=0.7)
            ax1.add_patch(rect)
            
            # Add block ID
            x_center, y_center = geo_data.centroids[block_id]
            ax1.text(x_center, y_center, block_id.replace('BLK', ''), 
                    ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Mark root blocks
            if block_id == root:
                ax1.scatter(x_center, y_center, s=200, c='red', marker='*', 
                           edgecolors='black', linewidth=2, label='District Root' if district_idx == 0 else "")
    
    # Mark depot location
    depot_x, depot_y = geo_data.centroids[depot_block]
    ax1.scatter(depot_x, depot_y, s=300, c='gold', marker='D', 
               edgecolors='black', linewidth=3, label='Depot')
    
    ax1.set_xlim(0, geo_data.region_size)
    ax1.set_ylim(0, geo_data.region_size)
    ax1.set_xlabel('Miles')
    ax1.set_ylabel('Miles')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: ODD Features Heatmap (Population Density)
    ax2 = axes[0, 1]
    ax2.set_title('Population Density (Ω_j[0])', fontsize=14, fontweight='bold')
    
    pop_densities = [Omega_dict[block_id][0] if hasattr(Omega_dict[block_id], '__len__') 
                    else Omega_dict[block_id] for block_id in geo_data.short_geoid_list]
    vmin, vmax = min(pop_densities), max(pop_densities)
    
    for i, block_id in enumerate(geo_data.short_geoid_list):
        x_min, y_min, x_max, y_max = geo_data.boundaries[block_id]
        omega_features = Omega_dict[block_id]
        pop_density = omega_features[0] if hasattr(omega_features, '__len__') else omega_features
        
        # Color intensity based on population density
        intensity = (pop_density - vmin) / (vmax - vmin) if vmax > vmin else 0
        color = plt.cm.Blues(intensity)
        
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.8)
        ax2.add_patch(rect)
        
        # Add population density text
        x_center, y_center = geo_data.centroids[block_id]
        ax2.text(x_center, y_center, f'{pop_density:.1f}', 
                ha='center', va='center', fontsize=9, fontweight='bold')
    
    ax2.set_xlim(0, geo_data.region_size)
    ax2.set_ylim(0, geo_data.region_size)
    ax2.set_xlabel('Miles')
    ax2.set_ylabel('Miles')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax2)
    cbar.set_label('Population Density')
    
    # Plot 3: District Costs Breakdown
    ax3 = axes[1, 0]
    ax3.set_title('District Costs Breakdown', fontsize=14, fontweight='bold')
    
    district_names = [f'District {i+1}\n(Root: {root.replace("BLK", "")})' 
                     for i, root in enumerate(districts.keys())]
    district_indices = [geo_data.short_geoid_list.index(root) for root in districts.keys()]
    
    K_district = [K_costs[idx] for idx in district_indices]
    F_district = [F_costs[idx] for idx in district_indices]
    
    # Handle 2D ODD vectors for display
    omega_district = []
    for idx in district_indices:
        omega_val = omega_values[idx]
        if hasattr(omega_val, '__len__') and len(omega_val) >= 2:
            omega_district.append(f"[{omega_val[0]:.1f}, {omega_val[1]:.1f}]")
        else:
            omega_district.append(f"{omega_val:.1f}")
    
    x_pos = np.arange(len(district_names))
    width = 0.35
    
    bars1 = ax3.bar(x_pos - width/2, K_district, width, label='Linehaul Cost (K_i)', 
                   color='skyblue', alpha=0.8)
    bars2 = ax3.bar(x_pos + width/2, F_district, width, label='ODD Cost (F_i)', 
                   color='lightcoral', alpha=0.8)
    
    ax3.set_xlabel('Districts')
    ax3.set_ylabel('Cost ($)')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(district_names)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${height:.1f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'${height:.1f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Convergence History
    ax4 = axes[1, 1]
    ax4.set_title('LBBD Convergence', fontsize=14, fontweight='bold')
    
    if 'history' in result and result['history']:
        iterations = [h['iteration'] for h in result['history']]
        lower_bounds = [h['lower_bound'] for h in result['history']]
        upper_bounds = [h['upper_bound'] for h in result['history']]
        gaps = [h['gap'] for h in result['history']]
        
        ax4_twin = ax4.twinx()
        
        line1 = ax4.plot(iterations, lower_bounds, 'b-o', label='Lower Bound', linewidth=2)
        line2 = ax4.plot(iterations, upper_bounds, 'r-s', label='Upper Bound', linewidth=2)
        line3 = ax4_twin.plot(iterations, gaps, 'g-^', label='Gap', linewidth=2, color='green')
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Objective Value', color='black')
        ax4_twin.set_ylabel('Gap', color='green')
        ax4.grid(True, alpha=0.3)
        
        # Combine legends
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('lbbd_service_region_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def run_service_region_test():
    """
    Main function to test LBBD on 10×10 mile service region
    """
    print("="*60)
    print("LBBD TEST: 10×10 Mile Service Region (5×5 Blocks)")
    print("="*60)
    
    # Create service region
    print("\n1. Creating service region...")
    geo_data = ServiceRegionGeoData(grid_size=5, region_size=10.0, seed=42)
    print(f"   - Region: {geo_data.region_size}×{geo_data.region_size} miles")
    print(f"   - Blocks: {geo_data.grid_size}×{geo_data.grid_size} grid ({geo_data.n_blocks} total)")
    print(f"   - Block size: {geo_data.block_size}×{geo_data.block_size} miles each")
    
    # Create ODD parameters
    print("\n2. Setting up ODD parameters...")
    Omega_dict, J_function = create_odd_parameters(geo_data)
    print(f"   - ODD features: 2D (population density, commercial activity)")
    print(f"   - ODD summary: ω_i = elementwise_max(Ω_j) for j ∈ district i")
    print(f"   - ODD cost: J(ω) = 2.0 + 1.2×ω[0] + 0.8×ω[1] + 0.1×ω[0]×ω[1]")
    
    # Show ODD feature ranges
    pop_densities = [omega[0] for omega in Omega_dict.values()]
    commercial_vals = [omega[1] for omega in Omega_dict.values()]
    print(f"   - Population density range: [{min(pop_densities):.2f}, {max(pop_densities):.2f}]")
    print(f"   - Commercial activity range: [{min(commercial_vals):.2f}, {max(commercial_vals):.2f}]")
    
    # Create probability distribution
    print("\n3. Creating demand distribution...")
    prob_dict = create_probability_distribution(geo_data)
    print(f"   - Based on population density and block area")
    print(f"   - Probability range: [{min(prob_dict.values()):.4f}, {max(prob_dict.values()):.4f}]")
    
    # Create partition instance
    print("\n4. Initializing LBBD...")
    partition = Partition(geo_data, num_districts=3, prob_dict=prob_dict, epsilon=0.5)
    print(f"   - Number of districts: 3")
    print(f"   - Wasserstein radius: 0.5 miles")
    
    # Run LBBD
    print("\n5. Running LBBD optimization...")
    print("   (This may take a few minutes...)")
    
    try:
        result = partition.benders_decomposition(
            max_iterations=10,
            tolerance=1e-2,
            max_cuts=50,
            verbose=True,
            Omega_dict=Omega_dict,
            J_function=J_function
        )
        
        print("\n" + "="*60)
        print("LBBD OPTIMIZATION RESULTS")
        print("="*60)
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iterations']}")
        print(f"Final gap: {result['final_gap']:.4f}")
        print(f"Best cost: ${result['best_cost']:.2f}")
        print(f"Optimal depot: {result['best_depot']}")
        
        # Show district details
        if result['best_partition'] is not None:
            z_sol = result['best_partition']
            districts = {}
            for i, block_id in enumerate(geo_data.short_geoid_list):
                assigned_to = np.argmax(z_sol[i])
                root_block = geo_data.short_geoid_list[assigned_to]
                if root_block not in districts:
                    districts[root_block] = []
                districts[root_block].append(block_id)
            
            print(f"\nDistrict composition:")
            for i, (root, blocks) in enumerate(districts.items()):
                root_idx = geo_data.short_geoid_list.index(root)
                K_i = result['best_K'][root_idx]
                F_i = result['best_F'][root_idx]
                omega_i = result['best_omega'][root_idx]
                print(f"  District {i+1} (Root: {root}):")
                print(f"    - Blocks: {[b.replace('BLK', '') for b in blocks]}")
                print(f"    - Linehaul cost (K_i): ${K_i:.2f}")
                print(f"    - ODD cost (F_i): ${F_i:.2f}")
                
                if hasattr(omega_i, '__len__') and len(omega_i) >= 2:
                    print(f"    - ODD summary (ω_i): [{omega_i[0]:.2f}, {omega_i[1]:.2f}] (pop_density, commercial)")
                else:
                    print(f"    - ODD summary (ω_i): {omega_i:.2f}")
                
                # Show the actual blocks in this district and their ODD features
                print(f"    - Block ODD features:")
                for block_id in blocks:
                    features = Omega_dict[block_id]
                    if hasattr(features, '__len__'):
                        print(f"      {block_id}: [{features[0]:.2f}, {features[1]:.2f}]")
                    else:
                        print(f"      {block_id}: {features:.2f}")
        
        # Visualize results
        print("\n6. Creating visualizations...")
        fig = visualize_results(geo_data, result, prob_dict, Omega_dict)
        print("   - Saved as 'lbbd_service_region_results.png'")
        
        print("\n✅ Service region test completed successfully!")
        return result, geo_data
        
    except Exception as e:
        print(f"\n❌ Error in LBBD optimization: {e}")
        import traceback
        traceback.print_exc()
        return None, geo_data


if __name__ == "__main__":
    result, geo_data = run_service_region_test()
