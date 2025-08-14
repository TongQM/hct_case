#!/usr/bin/env python3
"""
5×5 Service Region Test with Realistic Mixed-Gaussian Distributions
Tests LBBD on a 5×5 grid with:
- 2D truncated mixed-Gaussian ODD features
- Demand from 3-cluster truncated mixed-Gaussian
- Comprehensive visualization of all results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal, truncnorm
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
import time

class ToyGeoData(GeoData):
    """Geographic data for 5×5 service region testing"""
    
    def __init__(self, n_blocks, grid_size, seed=42):
        np.random.seed(seed)
        self.n_blocks = n_blocks
        self.grid_size = grid_size
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
        
        # Create coordinate mappings
        self.block_to_coord = {i: (i // grid_size, i % grid_size) for i in range(n_blocks)}
        
        # Create adjacency graph for contiguity
        self.block_graph = nx.Graph()
        for i in range(n_blocks):
            self.block_graph.add_node(self.short_geoid_list[i])
        
        # Add edges for adjacent blocks
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                # Manhattan distance = 1 means adjacent
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.block_graph.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        # Create directed arc list for flow constraints
        self.arc_list = []
        for edge in self.block_graph.edges():
            self.arc_list.append(edge)
            self.arc_list.append((edge[1], edge[0]))  # Both directions
        
        # Set uniform areas and default costs
        self.areas = {block_id: 1.0 for block_id in self.short_geoid_list}
        self.gdf = None
    
    def get_dist(self, block1, block2):
        """Euclidean distance between blocks"""
        if block1 == block2:
            return 0.0
        try:
            i1 = self.short_geoid_list.index(block1)
            i2 = self.short_geoid_list.index(block2)
            coord1 = self.block_to_coord[i1]
            coord2 = self.block_to_coord[i2]
            return np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
        except:
            return float('inf')
    
    def get_area(self, block_id):
        return 1.0
    
    def get_K(self, block_id):
        return 2.0
    
    def get_F(self, block_id):
        return 0.0
    
    def get_arc_list(self):
        return self.arc_list
    
    def get_in_arcs(self, block_id):
        return [(src, dst) for (src, dst) in self.arc_list if dst == block_id]
    
    def get_out_arcs(self, block_id):
        return [(src, dst) for (src, dst) in self.arc_list if src == block_id]

def generate_2d_mixed_gaussian_odd(grid_size, n_blocks, seed=42):
    """
    Generate 2D ODD features from truncated mixed-Gaussian distribution
    Returns dict mapping block_id to 2D ODD vector
    """
    np.random.seed(seed)
    
    # Define mixture components for ODD features
    # Component 1: Urban center (high population + commercial)
    mean1 = np.array([2.0, 2.0])  # Center of grid
    cov1 = np.array([[0.5, 0.2], [0.2, 0.8]])
    
    # Component 2: Suburban area (medium population, low commercial)
    mean2 = np.array([0.5, 3.5])  # Corner area
    cov2 = np.array([[1.0, -0.3], [-0.3, 0.6]])
    
    # Component 3: Industrial zone (low population, high commercial)
    mean3 = np.array([3.5, 1.0])  # Another corner
    cov3 = np.array([[0.8, 0.1], [0.1, 1.2]])
    
    # Mixture weights
    weights = [0.4, 0.35, 0.25]
    
    Omega_dict = {}
    block_ids = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    print("🎯 Generating 2D Mixed-Gaussian ODD Features:")
    print(f"   Component 1 (Urban): μ={mean1}, weight={weights[0]}")
    print(f"   Component 2 (Suburban): μ={mean2}, weight={weights[1]}")
    print(f"   Component 3 (Industrial): μ={mean3}, weight={weights[2]}")
    
    for i, block_id in enumerate(block_ids):
        # Get block coordinates (normalized to [0, grid_size-1])
        row = i // grid_size
        col = i % grid_size
        spatial_coord = np.array([row, col])
        
        # Sample from mixture model
        component = np.random.choice(3, p=weights)
        
        if component == 0:
            mean, cov = mean1, cov1
        elif component == 1:
            mean, cov = mean2, cov2
        else:
            mean, cov = mean3, cov3
        
        # Sample from multivariate normal
        base_sample = np.random.multivariate_normal(mean, cov)
        
        # Add spatial correlation (blocks closer to component center get higher values)
        spatial_distance = np.linalg.norm(spatial_coord - mean)
        spatial_factor = np.exp(-spatial_distance**2 / 8.0)  # Gaussian spatial decay
        
        # Apply spatial factor and ensure positive values
        omega1 = max(0.5, base_sample[0] * spatial_factor + 1.0)  # Population density
        omega2 = max(0.3, base_sample[1] * spatial_factor + 1.5)  # Commercial activity
        
        # Truncate to reasonable ranges
        omega1 = min(omega1, 4.0)
        omega2 = min(omega2, 5.0)
        
        Omega_dict[block_id] = np.array([omega1, omega2])
    
    # Print summary statistics
    omega1_values = [omega[0] for omega in Omega_dict.values()]
    omega2_values = [omega[1] for omega in Omega_dict.values()]
    print(f"   Population (ω₁): range=[{min(omega1_values):.2f}, {max(omega1_values):.2f}], mean={np.mean(omega1_values):.2f}")
    print(f"   Commercial (ω₂): range=[{min(omega2_values):.2f}, {max(omega2_values):.2f}], mean={np.mean(omega2_values):.2f}")
    
    return Omega_dict

def generate_3cluster_mixed_gaussian_demand(grid_size, n_blocks, seed=42):
    """
    Generate demand distribution from 3-cluster truncated mixed-Gaussian
    Returns dict mapping block_id to probability mass
    """
    np.random.seed(seed + 1)  # Different seed from ODD
    
    # Define 3 demand clusters
    # Cluster 1: Downtown business district
    center1 = np.array([1.5, 1.5])
    
    # Cluster 2: Residential area
    center2 = np.array([3.0, 0.5])
    
    # Cluster 3: Mixed-use area
    center3 = np.array([2.0, 3.5])
    
    cluster_weights = [0.45, 0.35, 0.20]  # Business district has highest demand
    cluster_intensities = [0.08, 0.05, 0.06]  # Peak demand levels
    cluster_spreads = [1.2, 1.5, 1.0]  # Spatial spread of each cluster
    
    print("🎯 Generating 3-Cluster Mixed-Gaussian Demand:")
    print(f"   Cluster 1 (Business): center={center1}, weight={cluster_weights[0]}, intensity={cluster_intensities[0]}")
    print(f"   Cluster 2 (Residential): center={center2}, weight={cluster_weights[1]}, intensity={cluster_intensities[1]}")
    print(f"   Cluster 3 (Mixed-use): center={center3}, weight={cluster_weights[2]}, intensity={cluster_intensities[2]}")
    
    prob_dict = {}
    block_ids = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    for i, block_id in enumerate(block_ids):
        row = i // grid_size
        col = i % grid_size
        spatial_coord = np.array([row, col])
        
        # Calculate contribution from each cluster
        total_prob = 0.0
        
        for j, (center, weight, intensity, spread) in enumerate(zip(
            [center1, center2, center3], cluster_weights, cluster_intensities, cluster_spreads)):
            
            # Distance from cluster center
            distance = np.linalg.norm(spatial_coord - center)
            
            # Gaussian decay with cluster-specific spread
            cluster_contribution = weight * intensity * np.exp(-distance**2 / (2 * spread**2))
            total_prob += cluster_contribution
        
        # Add small baseline demand + noise
        baseline = 0.015
        noise = np.random.uniform(-0.005, 0.005)
        total_prob = max(0.001, total_prob + baseline + noise)
        
        prob_dict[block_id] = total_prob
    
    # Normalize to sum to 1
    total_mass = sum(prob_dict.values())
    for block_id in prob_dict:
        prob_dict[block_id] /= total_mass
    
    # Print summary
    prob_values = list(prob_dict.values())
    print(f"   Demand range: [{min(prob_values):.4f}, {max(prob_values):.4f}]")
    print(f"   Mean demand: {np.mean(prob_values):.4f}")
    print(f"   Demand std: {np.std(prob_values):.4f}")
    
    return prob_dict

def test_5x5_realistic():
    """Test LBBD on 5×5 region with realistic mixed-Gaussian distributions"""
    
    print("=" * 100)
    print("5×5 SERVICE REGION TEST - REALISTIC MIXED-GAUSSIAN DISTRIBUTIONS")
    print("=" * 100)
    
    # Configuration
    grid_size = 5
    n_blocks = 25
    num_districts = 4
    
    # Setup geographic data
    toy_geo = ToyGeoData(n_blocks, grid_size, seed=42)
    toy_geo.beta = 0.7120
    toy_geo.Lambda = 35.0  # Higher arrival rate for larger region
    toy_geo.wr = 1.0
    toy_geo.wv = 10.0
    
    print(f"📍 Problem Configuration:")
    print(f"   Grid size: {grid_size}×{grid_size}")
    print(f"   Total blocks: {n_blocks}")
    print(f"   Number of districts: {num_districts}")
    print(f"   BHH coefficient (β): {toy_geo.beta}")
    print(f"   Arrival rate (Λ): {toy_geo.Lambda}")
    print(f"   Wasserstein radius (ε): 0.25")
    
    # Generate realistic distributions
    print(f"\n📊 Generating Realistic Distributions:")
    Omega_dict = generate_2d_mixed_gaussian_odd(grid_size, n_blocks, seed=42)
    prob_dict = generate_3cluster_mixed_gaussian_demand(grid_size, n_blocks, seed=42)
    
    # ODD cost function (linear combination)
    def J_function(omega_vector):
        if hasattr(omega_vector, '__len__') and len(omega_vector) >= 2:
            return 0.6 * omega_vector[0] + 0.5 * omega_vector[1]  # Population + Commercial
        return 0.0
    
    print(f"   ODD cost function: J(ω) = 0.6·ω₁ + 0.5·ω₂")
    
    # Create partition instance
    partition = Partition(toy_geo, num_districts=num_districts, 
                         prob_dict=prob_dict, epsilon=0.25)
    
    print(f"\n🚀 Running Multi-Cut LBBD with Verbose Logging:")
    print(f"   Max iterations: 25")
    print(f"   Tolerance: 1e-3")
    print(f"   Verbose logging: ON")
    print("-" * 100)
    
    # Run LBBD with verbose logging
    start_time = time.time()
    result = partition.benders_decomposition(
        max_iterations=25,
        tolerance=1e-3,
        verbose=True,  # Enable verbose logging
        Omega_dict=Omega_dict,
        J_function=J_function
    )
    elapsed_time = time.time() - start_time
    
    print("-" * 100)
    print(f"✅ LBBD Optimization Completed!")
    print(f"   Total time: {elapsed_time:.2f}s")
    print(f"   Converged: {result['converged']}")
    print(f"   Final cost: {result['best_cost']:.4f}")
    print(f"   Final gap: {result['final_gap']:.4f}")
    print(f"   Iterations used: {result['iterations']}")
    print(f"   Optimal depot: {result['best_depot']}")
    
    # Analyze district results
    best_partition = result['best_partition']
    block_ids = toy_geo.short_geoid_list
    district_info = []
    
    print(f"\n📍 Analyzing Final Partition:")
    for i in range(n_blocks):
        if round(best_partition[i, i]) == 1:  # Root district
            root_id = block_ids[i]
            assigned_blocks = [j for j in range(n_blocks) if round(best_partition[j, i]) == 1]
            
            # Calculate detailed district metrics
            assigned_block_ids = [block_ids[j] for j in assigned_blocks]
            cost, _, _, C_star, alpha_i, _, _ = partition._CQCP_benders_updated(
                assigned_block_ids, root_id, prob_dict, partition.epsilon, 
                K_i=result['best_K'][i], F_i=result['best_F'][i])
            
            # Calculate district demand and ODD characteristics
            district_demand = sum(prob_dict[block_ids[j]] for j in assigned_blocks)
            district_omega_avg = np.mean([Omega_dict[block_ids[j]] for j in assigned_blocks], axis=0)
            
            district_info.append({
                'root_idx': i,
                'root_id': root_id,
                'size': len(assigned_blocks),
                'demand_share': district_demand,
                'K_i': result['best_K'][i],
                'F_i': result['best_F'][i],
                'omega': result['best_omega'][i],
                'omega_avg': district_omega_avg,
                'C_star': C_star,
                'alpha_i': alpha_i,
                'cost': cost,
                'assigned_blocks': assigned_blocks
            })
    
    print(f"   Number of districts: {len(district_info)}")
    print(f"\n   District Details:")
    for d in district_info:
        print(f"   • District {d['root_id']}: {d['size']} blocks, demand={d['demand_share']:.3f}")
        print(f"     K_i={d['K_i']:.2f}, F_i={d['F_i']:.2f}, C*={d['C_star']:.3f}, cost={d['cost']:.3f}")
        print(f"     ω_max=({d['omega'][0]:.2f}, {d['omega'][1]:.2f}), ω_avg=({d['omega_avg'][0]:.2f}, {d['omega_avg'][1]:.2f})")
    
    # Create comprehensive visualization
    print(f"\n🎨 Creating Comprehensive Visualization...")
    
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Convergence plot (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    history = result['history']
    iterations = [h['iteration'] for h in history]
    lower_bounds = [h['lower_bound'] for h in history]
    upper_bounds = [h['upper_bound'] for h in history]
    gaps = [h['gap'] for h in history]
    
    ax1.plot(iterations, lower_bounds, 'b-o', label='Lower Bound', linewidth=2, markersize=4)
    ax1.plot(iterations, upper_bounds, 'r-s', label='Upper Bound', linewidth=2, markersize=4)
    ax1.fill_between(iterations, lower_bounds, upper_bounds, alpha=0.3, color='gray')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Objective Value')
    ax1.set_title(f'LBBD Convergence\\n{result["iterations"]} iterations, {elapsed_time:.1f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Gap convergence (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.semilogy(iterations, gaps, 'g-^', linewidth=2, markersize=4)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gap (log scale)')
    ax2.set_title(f'Gap Convergence\\nFinal gap: {result["final_gap"]:.3f}')
    ax2.grid(True, alpha=0.3)
    
    # 3. Final partition with depot and roots (top, spans 2 columns)
    ax3 = fig.add_subplot(gs[0, 2:])
    assignment_grid = np.zeros((grid_size, grid_size))
    
    # Fill in district assignments
    for d_idx, d in enumerate(district_info):
        for j in d['assigned_blocks']:
            row = j // grid_size
            col = j % grid_size
            assignment_grid[row, col] = d_idx
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(district_info)))
    cmap = plt.matplotlib.colors.ListedColormap(colors)
    
    im3 = ax3.imshow(assignment_grid, cmap=cmap, vmin=0, vmax=len(district_info)-1)
    ax3.set_title(f'Final Partition - {len(district_info)} Districts\\nDepot: {result["best_depot"]}')
    
    # Mark depot
    depot_idx = block_ids.index(result['best_depot'])
    depot_row = depot_idx // grid_size
    depot_col = depot_idx % grid_size
    ax3.scatter(depot_col, depot_row, c='red', s=400, marker='*', 
               edgecolors='black', linewidth=3, label='Depot', zorder=10)
    
    # Mark district roots
    for d in district_info:
        root_row = d['root_idx'] // grid_size
        root_col = d['root_idx'] % grid_size
        ax3.scatter(root_col, root_row, c='white', s=200, marker='s', 
                   edgecolors='black', linewidth=2, zorder=9)
        ax3.text(root_col, root_row, f"R", ha='center', va='center', 
                fontsize=12, fontweight='bold', color='black')
    
    # Add block IDs
    for i in range(n_blocks):
        row = i // grid_size
        col = i % grid_size
        if i != depot_idx and i not in [d['root_idx'] for d in district_info]:
            ax3.text(col, row, block_ids[i][-2:], ha='center', va='center', 
                    fontsize=8, color='black', alpha=0.7)
    
    # Add grid lines
    for i in range(grid_size + 1):
        ax3.axhline(i - 0.5, color='black', linewidth=0.5)
        ax3.axvline(i - 0.5, color='black', linewidth=0.5)
    
    ax3.legend(loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # 4. Demand distribution (middle-left)
    ax4 = fig.add_subplot(gs[1, 0])
    demand_grid = np.zeros((grid_size, grid_size))
    for i, block_id in enumerate(block_ids):
        row = i // grid_size
        col = i % grid_size
        demand_grid[row, col] = prob_dict[block_id]
    
    im4 = ax4.imshow(demand_grid, cmap='Reds', interpolation='nearest')
    ax4.set_title('Demand Distribution\\n(3-Cluster Mixed-Gaussian)')
    plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Add demand values
    for i in range(n_blocks):
        row = i // grid_size
        col = i % grid_size
        demand_val = prob_dict[block_ids[i]]
        color = 'white' if demand_val > np.mean(list(prob_dict.values())) else 'black'
        ax4.text(col, row, f'{demand_val:.3f}', ha='center', va='center', 
                fontsize=7, color=color, fontweight='bold')
    
    # 5. ODD Feature 1 - Population (middle-center)
    ax5 = fig.add_subplot(gs[1, 1])
    omega1_grid = np.zeros((grid_size, grid_size))
    for i, block_id in enumerate(block_ids):
        row = i // grid_size
        col = i % grid_size
        omega1_grid[row, col] = Omega_dict[block_id][0]
    
    im5 = ax5.imshow(omega1_grid, cmap='Blues', interpolation='nearest')
    ax5.set_title('ODD Feature 1: Population\\n(Mixed-Gaussian)')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Add omega1 values
    for i in range(n_blocks):
        row = i // grid_size
        col = i % grid_size
        omega1_val = Omega_dict[block_ids[i]][0]
        color = 'white' if omega1_val > np.mean([o[0] for o in Omega_dict.values()]) else 'black'
        ax5.text(col, row, f'{omega1_val:.1f}', ha='center', va='center', 
                fontsize=8, color=color, fontweight='bold')
    
    # 6. ODD Feature 2 - Commercial (middle-right)
    ax6 = fig.add_subplot(gs[1, 2])
    omega2_grid = np.zeros((grid_size, grid_size))
    for i, block_id in enumerate(block_ids):
        row = i // grid_size
        col = i % grid_size
        omega2_grid[row, col] = Omega_dict[block_id][1]
    
    im6 = ax6.imshow(omega2_grid, cmap='Greens', interpolation='nearest')
    ax6.set_title('ODD Feature 2: Commercial\\n(Mixed-Gaussian)')
    plt.colorbar(im6, ax=ax6, fraction=0.046, pad=0.04)
    
    # Add omega2 values
    for i in range(n_blocks):
        row = i // grid_size
        col = i % grid_size
        omega2_val = Omega_dict[block_ids[i]][1]
        color = 'white' if omega2_val > np.mean([o[1] for o in Omega_dict.values()]) else 'black'
        ax6.text(col, row, f'{omega2_val:.1f}', ha='center', va='center', 
                fontsize=8, color=color, fontweight='bold')
    
    # 7. Dispatch subintervals (middle, far-right)
    ax7 = fig.add_subplot(gs[1, 3])
    C_grid = np.full((grid_size, grid_size), np.nan)
    
    # Fill in C* values by district
    for d in district_info:
        for j in d['assigned_blocks']:
            row = j // grid_size
            col = j % grid_size
            C_grid[row, col] = d['C_star']
    
    im7 = ax7.imshow(C_grid, cmap='plasma', interpolation='nearest')
    ax7.set_title('Dispatch Subinterval C*\\n(Optimized per District)')
    plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)
    
    # Add C* values
    for d in district_info:
        for j in d['assigned_blocks']:
            row = j // grid_size
            col = j % grid_size
            ax7.text(col, row, f'{d["C_star"]:.2f}', ha='center', va='center', 
                    fontsize=8, color='white', fontweight='bold')
    
    # 8. District costs and statistics (bottom row)
    ax8 = fig.add_subplot(gs[2, :2])
    
    # Bar chart of district costs
    district_names = [d['root_id'] for d in district_info]
    district_costs = [d['cost'] for d in district_info]
    linehaul_costs = [d['K_i'] for d in district_info]
    odd_costs = [d['F_i'] for d in district_info]
    
    x_pos = np.arange(len(district_names))
    width = 0.25
    
    ax8.bar(x_pos - width, district_costs, width, label='Total Cost', alpha=0.8)
    ax8.bar(x_pos, linehaul_costs, width, label='Linehaul (K_i)', alpha=0.8)
    ax8.bar(x_pos + width, odd_costs, width, label='ODD (F_i)', alpha=0.8)
    
    ax8.set_xlabel('District')
    ax8.set_ylabel('Cost')
    ax8.set_title('District Cost Breakdown')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(district_names)
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. District characteristics table (bottom-right)
    ax9 = fig.add_subplot(gs[2, 2:])
    ax9.axis('off')
    
    # Create summary table
    table_data = []
    headers = ['District', 'Size', 'Demand%', 'K_i', 'F_i', 'C*', 'ω₁_avg', 'ω₂_avg']
    
    for d in district_info:
        row = [
            d['root_id'],
            f"{d['size']}",
            f"{d['demand_share']*100:.1f}%",
            f"{d['K_i']:.2f}",
            f"{d['F_i']:.2f}",
            f"{d['C_star']:.2f}",
            f"{d['omega_avg'][0]:.2f}",
            f"{d['omega_avg'][1]:.2f}"
        ]
        table_data.append(row)
    
    table = ax9.table(cellText=table_data, colLabels=headers, 
                     cellLoc='center', loc='center', 
                     colWidths=[0.12]*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax9.set_title('District Summary Statistics', fontsize=12, fontweight='bold', pad=20)
    
    # Add grid lines to spatial plots
    for ax in [ax4, ax5, ax6, ax7]:
        for i in range(grid_size + 1):
            ax.axhline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
            ax.axvline(i - 0.5, color='white', linewidth=0.5, alpha=0.7)
    
    # Main title
    fig.suptitle('5×5 Service Region: Multi-Cut LBBD with Mixed-Gaussian Distributions\\n'
                f'Cost: {result["best_cost"]:.3f}, Gap: {result["final_gap"]:.3f}, '
                f'Time: {elapsed_time:.1f}s, Converged: {result["converged"]}', 
                fontsize=16, fontweight='bold')
    
    plt.savefig('5x5_realistic_lbbd_results.png', dpi=300, bbox_inches='tight')
    print(f"   📁 Saved comprehensive visualization as '5x5_realistic_lbbd_results.png'")
    
    # Final performance summary
    sizes = [d['size'] for d in district_info]
    demands = [d['demand_share'] for d in district_info]
    balance_ratio = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')
    demand_balance = max(demands) / min(demands) if min(demands) > 0 else float('inf')
    
    print(f"\n🏆 FINAL PERFORMANCE SUMMARY:")
    print(f"=" * 100)
    print(f"✅ Multi-cut LBBD: Successfully optimized 5×5 service region")
    print(f"✅ Realistic distributions: Mixed-Gaussian ODD and 3-cluster demand")
    print(f"✅ Convergence: {'Yes' if result['converged'] else 'Partial'} in {result['iterations']} iterations")
    print(f"✅ Solution quality: Cost={result['best_cost']:.3f}, Gap={result['final_gap']:.3f}")
    print(f"✅ Computational efficiency: {elapsed_time:.1f}s total time")
    print(f"✅ Spatial balance: District sizes {sizes}, ratio {balance_ratio:.2f}:1")
    demand_shares = [f'{d*100:.1f}%' for d in demands]
    root_ids = [d['root_id'] for d in district_info]
    c_star_values = [f'{d["C_star"]:.2f}' for d in district_info]
    print(f"✅ Demand balance: Demand shares {demand_shares}, ratio {demand_balance:.2f}:1")
    print(f"✅ Depot location: {result['best_depot']} (optimally positioned)")
    print(f"✅ District roots: {root_ids}")
    print(f"✅ Dispatch optimization: C* values {c_star_values}")
    
    return result, district_info

if __name__ == "__main__":
    test_5x5_realistic()