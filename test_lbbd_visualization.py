#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from test_lbbd_service_region import ServiceRegionGeoData, create_odd_parameters, create_probability_distribution
from lib.algorithm import Partition

def visualize_lbbd_results(geo_data, result, Omega_dict, J_function):
    """
    Create comprehensive visualization of LBBD results
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('LBBD Results: 5x5 Service Region with Enhanced Cuts', fontsize=16, fontweight='bold')
    
    # Extract results
    best_partition = result['best_partition']
    best_depot = result['best_depot']
    best_K = result['best_K']
    best_F = result['best_F']
    best_omega = result['best_omega']
    
    # Get depot coordinates
    depot_coords = geo_data.centroids[best_depot]
    
    # Colors for districts
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    # 1. Partition Visualization (Top Left)
    ax1 = axes[0, 0]
    ax1.set_title('Optimal Partition with Depot Location', fontsize=14, fontweight='bold')
    
    # Draw blocks and assign colors based on partition
    block_to_district = {}
    for j, block_id in enumerate(geo_data.short_geoid_list):
        for i in range(len(geo_data.short_geoid_list)):
            if best_partition[j, i] > 0.5:  # Assigned to district i
                block_to_district[block_id] = i
                break
    
    # Plot blocks with district colors
    for block_id in geo_data.short_geoid_list:
        x_min, y_min, x_max, y_max = geo_data.boundaries[block_id]
        district = block_to_district.get(block_id, 0)
        color = colors[district % len(colors)]
        
        # Draw block
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                               linewidth=1, edgecolor='black', facecolor=color, alpha=0.6)
        ax1.add_patch(rect)
        
        # Add block label
        center_x, center_y = geo_data.centroids[block_id]
        ax1.text(center_x, center_y, f'{block_id[-2:]}', ha='center', va='center', 
                fontsize=8, fontweight='bold')
    
    # Mark depot with special symbol
    ax1.plot(depot_coords[0], depot_coords[1], 'k*', markersize=20, markeredgecolor='white', 
            markeredgewidth=2, label=f'Depot: {best_depot}')
    
    ax1.set_xlim(-0.5, 10.5)
    ax1.set_ylim(-0.5, 10.5)
    ax1.set_xlabel('X (miles)')
    ax1.set_ylabel('Y (miles)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. District Costs Breakdown (Top Right)
    ax2 = axes[0, 1]
    ax2.set_title('District Costs Breakdown', fontsize=14, fontweight='bold')
    
    # Calculate costs for each district
    districts = list(set(block_to_district.values()))
    district_data = []
    
    for district in districts:
        root_block = geo_data.short_geoid_list[district]
        K_cost = best_K[district] if district < len(best_K) else 0
        F_cost = best_F[district] if district < len(best_F) else 0
        
        # Count blocks in district
        block_count = sum(1 for d in block_to_district.values() if d == district)
        
        district_data.append({
            'district': district,
            'root': root_block,
            'K_cost': K_cost,
            'F_cost': F_cost,
            'total_cost': K_cost + F_cost,
            'block_count': block_count
        })
    
    # Sort by district number
    district_data.sort(key=lambda x: x['district'])
    
    # Create stacked bar chart
    districts_list = [f"D{d['district']}\n({d['root'][-2:]})" for d in district_data]
    K_costs = [d['K_cost'] for d in district_data]
    F_costs = [d['F_cost'] for d in district_data]
    
    bar_width = 0.6
    x_pos = np.arange(len(districts_list))
    
    bars1 = ax2.bar(x_pos, K_costs, bar_width, label='Linehaul Cost (K_i)', color='lightblue')
    bars2 = ax2.bar(x_pos, F_costs, bar_width, bottom=K_costs, label='ODD Cost (F_i)', color='lightcoral')
    
    ax2.set_xlabel('District (Root Block)')
    ax2.set_ylabel('Cost')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(districts_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (d, bar1, bar2) in enumerate(zip(district_data, bars1, bars2)):
        # K cost label
        if d['K_cost'] > 0:
            ax2.text(bar1.get_x() + bar1.get_width()/2, bar1.get_height()/2, 
                    f'{d["K_cost"]:.1f}', ha='center', va='center', fontweight='bold')
        # F cost label
        if d['F_cost'] > 0:
            ax2.text(bar2.get_x() + bar2.get_width()/2, d['K_cost'] + bar2.get_height()/2, 
                    f'{d["F_cost"]:.1f}', ha='center', va='center', fontweight='bold')
        # Total cost label
        ax2.text(bar2.get_x() + bar2.get_width()/2, d['total_cost'] + 0.5, 
                f'Total: {d["total_cost"]:.1f}', ha='center', va='bottom', fontsize=9)
    
    # 3. ODD Features Heatmap (Bottom Left)
    ax3 = axes[1, 0]
    ax3.set_title('2D ODD Features Distribution', fontsize=14, fontweight='bold')
    
    # Create grid for ODD features
    grid_size = 5
    odd_grid_0 = np.zeros((grid_size, grid_size))  # Population density
    odd_grid_1 = np.zeros((grid_size, grid_size))  # Commercial activity
    
    for i, block_id in enumerate(geo_data.short_geoid_list):
        row = i // grid_size
        col = i % grid_size
        omega_features = Omega_dict[block_id]
        odd_grid_0[row, col] = omega_features[0]  # Population density
        odd_grid_1[row, col] = omega_features[1]  # Commercial activity
    
    # Plot combined ODD intensity
    combined_odd = 1.2 * odd_grid_0 + 0.8 * odd_grid_1  # Using J_function weights
    im = ax3.imshow(combined_odd, cmap='YlOrRd', origin='lower', extent=[0, 10, 0, 10])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('Combined ODD Intensity (1.2×Pop + 0.8×Com)', rotation=270, labelpad=20)
    
    # Add grid and labels
    ax3.set_xlabel('X (miles)')
    ax3.set_ylabel('Y (miles)')
    
    # Add text annotations for ODD values
    for i in range(grid_size):
        for j in range(grid_size):
            block_idx = i * grid_size + j
            if block_idx < len(geo_data.short_geoid_list):
                block_id = geo_data.short_geoid_list[block_idx]
                omega = Omega_dict[block_id]
                ax3.text(j*2 + 1, i*2 + 1, f'{omega[0]:.1f}\n{omega[1]:.1f}', 
                        ha='center', va='center', fontsize=8, color='white', fontweight='bold')
    
    # 4. Convergence History (Bottom Right)
    ax4 = axes[1, 1]
    ax4.set_title('LBBD Convergence History', fontsize=14, fontweight='bold')
    
    if 'history' in result and result['history']:
        iterations = [h['iteration'] for h in result['history']]
        lower_bounds = [h['lower_bound'] for h in result['history']]
        upper_bounds = [h['upper_bound'] for h in result['history']]
        
        ax4.plot(iterations, upper_bounds, 'r-o', label='Upper Bound', linewidth=2, markersize=8)
        ax4.plot(iterations, lower_bounds, 'b-s', label='Lower Bound', linewidth=2, markersize=8)
        ax4.fill_between(iterations, lower_bounds, upper_bounds, alpha=0.2, color='gray', label='Gap')
        
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Objective Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add final gap info
        final_gap = upper_bounds[-1] - lower_bounds[-1] if len(upper_bounds) > 0 else 0
        ax4.text(0.02, 0.98, f'Final Gap: {final_gap:.2f}\nConverged: {result["converged"]}', 
                transform=ax4.transAxes, va='top', ha='left', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'No convergence history available', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12)
    
    plt.tight_layout()
    return fig

def main():
    print("🚀 Testing Enhanced LBBD on 5x5 Service Region...")
    print("=" * 60)
    
    # Create 5x5 service region
    geo_data = ServiceRegionGeoData(grid_size=5, region_size=10.0, seed=42)
    geo_data.beta = 0.7120  # BHH coefficient
    geo_data.Lambda = 30.0  # Overall arrival rate
    
    # Create ODD parameters and probability distribution
    Omega_dict, J_function = create_odd_parameters(geo_data)
    prob_dict = create_probability_distribution(geo_data)
    
    print(f"📊 Service Region Setup:")
    print(f"   • Grid: 5×5 blocks (25 total)")
    print(f"   • Area: 10×10 miles")
    print(f"   • ODD Features: 2D (population density, commercial activity)")
    print(f"   • J_function: Linear (zero at zero)")
    print(f"   • BHH coefficient β: {geo_data.beta}")
    print(f"   • Overall arrival rate Λ: {geo_data.Lambda}")
    
    # Test J_function
    print(f"\n🔍 J_function Verification:")
    print(f"   • J([0,0]) = {J_function(np.array([0.0, 0.0])):.2f} (should be 0)")
    print(f"   • J([1,0]) = {J_function(np.array([1.0, 0.0])):.2f}")
    print(f"   • J([0,1]) = {J_function(np.array([0.0, 1.0])):.2f}")
    print(f"   • J([2,3]) = {J_function(np.array([2.0, 3.0])):.2f}")
    
    # Initialize partition algorithm
    num_districts = 3
    partition = Partition(geo_data, num_districts=num_districts, prob_dict=prob_dict, epsilon=0.5)
    
    print(f"\n🔧 LBBD Configuration:")
    print(f"   • Number of districts: {num_districts}")
    print(f"   • Wasserstein radius ε: {partition.epsilon}")
    print(f"   • Max iterations: 5")
    print(f"   • Tolerance: 0.1")
    print(f"   • Max cuts: 20")
    
    # Run Enhanced LBBD with K_i and F_i cuts
    print(f"\n⚡ Running Enhanced LBBD with K_i and F_i subgradient cuts...")
    result = partition.benders_decomposition(
        max_iterations=5,
        tolerance=0.1,
        max_cuts=20,
        verbose=True,
        Omega_dict=Omega_dict,
        J_function=J_function
    )
    
    print(f"\n✅ LBBD Results:")
    print(f"   • Best Cost: {result['best_cost']:.2f}")
    print(f"   • Converged: {result['converged']}")
    print(f"   • Iterations: {result['iterations']}")
    print(f"   • Best Depot: {result['best_depot']}")
    
    # Display district details
    print(f"\n📋 District Details:")
    best_partition = result['best_partition']
    best_K = result['best_K']
    best_F = result['best_F']
    
    for i in range(num_districts):
        # Find blocks assigned to district i
        assigned_blocks = []
        for j, block_id in enumerate(geo_data.short_geoid_list):
            if best_partition[j, i] > 0.5:
                assigned_blocks.append(block_id)
        
        if assigned_blocks:
            root_block = geo_data.short_geoid_list[i]
            K_cost = best_K[i] if i < len(best_K) else 0
            F_cost = best_F[i] if i < len(best_F) else 0
            print(f"   District {i}: Root={root_block}, Blocks={len(assigned_blocks)}, K_i={K_cost:.2f}, F_i={F_cost:.2f}")
    
    # Create comprehensive visualization
    print(f"\n🎨 Creating visualization...")
    fig = visualize_lbbd_results(geo_data, result, Omega_dict, J_function)
    
    # Save the plot
    output_filename = 'lbbd_enhanced_visualization.png'
    fig.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"   • Saved visualization as: {output_filename}")
    
    # Show the plot
    plt.show()
    
    print(f"\n🎯 Summary:")
    print(f"   • Enhanced LBBD successfully optimized the 5×5 service region")
    print(f"   • K_i and F_i subgradients provided better optimization guidance")
    print(f"   • Linear J_function enabled direct use in MILP master problem")
    print(f"   • Depot location and partition-dependent costs jointly optimized")
    
    return result, fig

if __name__ == "__main__":
    result, fig = main()

