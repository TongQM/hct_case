#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from test_lbbd_service_region import ServiceRegionGeoData, create_odd_parameters, create_probability_distribution
from lib.algorithm import Partition
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_balanced_odd_parameters(geo_data):
    """
    Create more balanced ODD parameters to encourage balanced partitions
    """
    # Each block has a 2D ODD feature vector [population_density, commercial_activity]
    Omega_dict = {}
    for block_id in geo_data.short_geoid_list:
        features = geo_data.get_odd_features(block_id)
        # Scale down the features to reduce ODD cost dominance
        scaled_features = features * 0.5  # Reduce ODD magnitude
        Omega_dict[block_id] = scaled_features
    
    # More balanced ODD cost function
    def J_function(omega_vector):
        """
        More balanced ODD cost function for 2D feature vector
        """
        if isinstance(omega_vector, (int, float)):
            # Handle scalar input (for piecewise linearization)
            return 1.0 + 0.8 * omega_vector  # Reduced coefficients
        
        omega1, omega2 = omega_vector[0], omega_vector[1]
        base_cost = 1.0  # Reduced base cost
        w1, w2 = 0.8, 0.6  # Reduced weights
        interaction = 0.05   # Reduced interaction
        return base_cost + w1 * omega1 + w2 * omega2 + interaction * omega1 * omega2
    
    return Omega_dict, J_function

def test_balanced_partition():
    """
    Test LBBD with parameters that encourage more balanced partitions
    """
    print("="*70)
    print("BALANCED PARTITION TEST: Adjusted Parameters for Better Balance")
    print("="*70)
    
    # Create service region
    print("\n1. Creating service region...")
    geo_data = ServiceRegionGeoData(grid_size=5, region_size=10.0, seed=42)
    print(f"   - Region: {geo_data.region_size}×{geo_data.region_size} miles")
    print(f"   - Blocks: {geo_data.grid_size}×{geo_data.grid_size} grid ({geo_data.n_blocks} total)")
    
    # Create balanced ODD parameters
    print("\n2. Setting up balanced ODD parameters...")
    Omega_dict, J_function = create_balanced_odd_parameters(geo_data)
    print(f"   - Reduced ODD feature magnitudes by 50%")
    print(f"   - Balanced cost function: J(ω) = 1.0 + 0.8×ω[0] + 0.6×ω[1] + 0.05×ω[0]×ω[1]")
    
    # Create probability distribution
    prob_dict = create_probability_distribution(geo_data)
    
    # Test with different numbers of districts
    for num_districts in [3, 4, 5]:
        print(f"\n3. Testing with {num_districts} districts...")
        partition = Partition(geo_data, num_districts=num_districts, prob_dict=prob_dict, epsilon=0.3)
        
        try:
            result = partition.benders_decomposition(
                max_iterations=8,
                tolerance=5e-2,  # Relaxed tolerance
                max_cuts=30,
                verbose=False,  # Reduce output
                Omega_dict=Omega_dict,
                J_function=J_function
            )
            
            print(f"\n   Results for {num_districts} districts:")
            print(f"   - Converged: {result['converged']}")
            print(f"   - Best cost: ${result['best_cost']:.2f}")
            print(f"   - Depot: {result['best_depot']}")
            
            # Analyze partition balance
            if result['best_partition'] is not None:
                z_sol = result['best_partition']
                districts = {}
                for i, block_id in enumerate(geo_data.short_geoid_list):
                    assigned_to = np.argmax(z_sol[i])
                    root_block = geo_data.short_geoid_list[assigned_to]
                    if root_block not in districts:
                        districts[root_block] = []
                    districts[root_block].append(block_id)
                
                district_sizes = [len(blocks) for blocks in districts.values()]
                print(f"   - District sizes: {district_sizes}")
                print(f"   - Balance ratio: {min(district_sizes)/max(district_sizes):.2f} (1.0 = perfect balance)")
                print(f"   - Std deviation: {np.std(district_sizes):.2f}")
                
                # Visualize the best balanced partition
                if num_districts == 4:  # Choose 4 districts for visualization
                    visualize_balanced_partition(geo_data, result, Omega_dict, num_districts)
                    
        except Exception as e:
            print(f"   - Error: {e}")
    
    print(f"\n✅ Balanced partition test completed!")

def visualize_balanced_partition(geo_data, result, Omega_dict, num_districts):
    """
    Visualize the balanced partition
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Balanced Partition Results: {num_districts} Districts', fontsize=16, fontweight='bold')
    
    # Define colors for districts
    district_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    # Extract partition information
    z_sol = result['best_partition']
    depot_block = result['best_depot']
    
    # Identify districts and their roots
    districts = {}
    for i, block_id in enumerate(geo_data.short_geoid_list):
        assigned_to = np.argmax(z_sol[i])
        root_block = geo_data.short_geoid_list[assigned_to]
        if root_block not in districts:
            districts[root_block] = []
        districts[root_block].append(block_id)
    
    # Plot 1: Partition visualization
    ax1 = axes[0]
    ax1.set_title('Balanced Partition', fontsize=14, fontweight='bold')
    
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
                           edgecolors='black', linewidth=2, 
                           label='District Root' if district_idx == 0 else "")
    
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
    
    # Plot 2: District size comparison
    ax2 = axes[1]
    ax2.set_title('District Size Comparison', fontsize=14, fontweight='bold')
    
    district_names = [f'District {i+1}\n(Root: {root.replace("BLK", "")})' 
                     for i, root in enumerate(districts.keys())]
    district_sizes = [len(blocks) for blocks in districts.values()]
    
    bars = ax2.bar(district_names, district_sizes, 
                  color=[district_colors[i % len(district_colors)] for i in range(len(district_names))],
                  alpha=0.7, edgecolor='black', linewidth=2)
    
    # Add value labels on bars
    for bar, size in zip(bars, district_sizes):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{size} blocks', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax2.set_ylabel('Number of Blocks')
    ax2.set_xlabel('Districts')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add balance metrics
    balance_ratio = min(district_sizes) / max(district_sizes)
    std_dev = np.std(district_sizes)
    ax2.text(0.02, 0.98, f'Balance Ratio: {balance_ratio:.2f}\nStd Deviation: {std_dev:.2f}', 
             transform=ax2.transAxes, verticalalignment='top', fontsize=12,
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'balanced_partition_{num_districts}districts.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_balanced_partition()

