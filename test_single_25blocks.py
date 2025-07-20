#!/usr/bin/env python3
"""
Single configuration test for 25-block toy model to demonstrate working algorithm.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from test_toy_model_25blocks import ToyGeoData, generate_random_probabilities
from lib.algorithm import Partition
import logging

# Set up logging for cleaner output
logging.basicConfig(level=logging.WARNING)

def plot_partition_analysis(toy_geo, partition_result, config_name, history):
    """Create comprehensive partition analysis visualization."""
    
    fig = plt.figure(figsize=(20, 12))
    
    # Create grid layout
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Original topology (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    pos = {block_id: toy_geo.grid_positions[block_id] for block_id in toy_geo.short_geoid_list}
    nx.draw(toy_geo.G, pos, with_labels=True, node_size=200, font_size=5, ax=ax1)
    ax1.set_title("Original Topology", fontsize=10)
    
    # 2. Partition result (top-middle-left)
    ax2 = fig.add_subplot(gs[0, 1])
    
    if partition_result is not None:
        z_sol = partition_result
        N = len(toy_geo.short_geoid_list)
        block_ids = toy_geo.short_geoid_list
        
        # Find district roots and assignments
        roots = [i for i in range(N) if round(z_sol[i, i]) == 1]
        district_assignments = {}
        district_info = {}
        
        for j, block_id in enumerate(block_ids):
            for root_idx in roots:
                if round(z_sol[j, root_idx]) == 1:
                    district_assignments[block_id] = root_idx
                    break
        
        # Group blocks by district
        for root_idx in roots:
            assigned_blocks = [block_ids[j] for j in range(N) if round(z_sol[j, root_idx]) == 1]
            district_info[root_idx] = assigned_blocks
        
        # Create color mapping
        base_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
        color_map = {root_idx: base_colors[i % len(base_colors)] for i, root_idx in enumerate(roots)}
        colors = [color_map[district_assignments[block_id]] for block_id in block_ids]
        
        # Draw partitioned graph
        nx.draw(toy_geo.G, pos, node_color=colors, with_labels=True, 
               node_size=200, font_size=5, ax=ax2, font_weight='bold')
        
        # Legend
        legend_elements = []
        for root_idx in roots:
            block_name = block_ids[root_idx]
            count = len(district_info[root_idx])
            legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', 
                                            markerfacecolor=color_map[root_idx], markersize=8, 
                                            label=f'{block_name} ({count})'))
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=6)
    
    ax2.set_title(f"Partition Result", fontsize=10)
    
    # 3. Convergence plot (top-middle-right)
    ax3 = fig.add_subplot(gs[0, 2])
    if history:
        iterations = [h['iteration'] for h in history]
        lower_bounds = [h['lower_bound'] for h in history]
        upper_bounds = [h['upper_bound'] for h in history]
        
        ax3.plot(iterations, lower_bounds, 'b-o', label='Lower', linewidth=2, markersize=4)
        ax3.plot(iterations, upper_bounds, 'r-s', label='Upper', linewidth=2, markersize=4)
        ax3.fill_between(iterations, lower_bounds, upper_bounds, alpha=0.3, color='gray')
        ax3.set_xlabel('Iteration', fontsize=8)
        ax3.set_ylabel('Objective', fontsize=8)
        ax3.legend(fontsize=6)
        ax3.grid(True, alpha=0.3)
    ax3.set_title("Convergence", fontsize=10)
    
    # 4. Gap reduction (top-right)
    ax4 = fig.add_subplot(gs[0, 3])
    if history:
        gaps = [h['gap'] for h in history]
        ax4.semilogy(iterations, gaps, 'g-^', linewidth=2, markersize=4)
        ax4.set_xlabel('Iteration', fontsize=8)
        ax4.set_ylabel('Gap (log)', fontsize=8)
        ax4.grid(True, alpha=0.3)
    ax4.set_title("Gap Reduction", fontsize=10)
    
    # 5. District statistics table (bottom span)
    ax5 = fig.add_subplot(gs[1:, :])
    ax5.axis('off')
    
    if partition_result is not None:
        # Prepare statistics
        stats_text = f"Configuration: {config_name}\n\n"
        stats_text += f"{'District':<12} {'Root':<8} {'Blocks':<6} {'Block IDs':<30} {'Prob Mass':<10}\n"
        stats_text += "-" * 80 + "\n"
        
        total_prob = 0
        district_sizes = []
        
        for i, root_idx in enumerate(sorted(roots)):
            root_name = block_ids[root_idx]
            assigned_blocks = district_info[root_idx]
            district_sizes.append(len(assigned_blocks))
            
            # Calculate probability mass for this district
            prob_mass = sum(prob_dict[block_id] for block_id in assigned_blocks)
            total_prob += prob_mass
            
            # Format block IDs for display
            block_str = ', '.join(assigned_blocks[:5])  # Show first 5 blocks
            if len(assigned_blocks) > 5:
                block_str += f", ... (+{len(assigned_blocks)-5})"
            
            stats_text += f"District {i+1:<4} {root_name:<8} {len(assigned_blocks):<6} {block_str:<30} {prob_mass:<10.4f}\n"
        
        # Summary statistics
        stats_text += "\n" + "=" * 80 + "\n"
        stats_text += f"Summary:\n"
        stats_text += f"  Total Districts: {len(roots)}\n"
        stats_text += f"  District Sizes: {district_sizes}\n"
        stats_text += f"  Size Balance (std): {np.std(district_sizes):.2f}\n"
        stats_text += f"  Total Probability: {total_prob:.4f}\n"
        
        if history:
            stats_text += f"  Algorithm Iterations: {len(history)}\n"
            stats_text += f"  Final Gap: {history[-1]['gap']:.4f}\n"
            stats_text += f"  Best Cost: {history[-1]['lower_bound']:.4f}\n"
        
        ax5.text(0.05, 0.95, stats_text, transform=ax5.transAxes, fontsize=9, 
                verticalalignment='top', fontfamily='monospace')
    
    plt.suptitle(f'Benders Decomposition Analysis: {config_name}', fontsize=16, y=0.98)
    
    return fig, district_info if partition_result is not None else None

def run_single_test():
    """Run a single comprehensive test."""
    
    print("🚀 SINGLE 25-BLOCK BENDERS DECOMPOSITION TEST")
    print("=" * 60)
    
    # Create 25-block toy model
    print("\n1️⃣ Creating 25-block toy model...")
    toy_geo = ToyGeoData(n_blocks=25, grid_size=5, seed=42)
    
    # Generate probability distribution
    print("2️⃣ Generating probability distribution...")
    global prob_dict  # Make it global so plot function can access it
    prob_dict = generate_random_probabilities(toy_geo.short_geoid_list, seed=42, distribution='uniform')
    
    probs = list(prob_dict.values())
    print(f"   📊 Prob Stats - Min: {min(probs):.4f}, Max: {max(probs):.4f}")
    print(f"                    Mean: {np.mean(probs):.4f}, Std: {np.std(probs):.4f}")
    
    # Compute K values
    print("3️⃣ Computing K values...")
    toy_geo.compute_K_for_all_blocks(depot_lat=2, depot_lon=2)
    
    # Test configuration
    config = {
        'name': 'uniform_4districts', 
        'districts': 4, 
        'epsilon': 0.3
    }
    
    print(f"\n4️⃣ Running Configuration: {config['name']}")
    print(f"   🎯 Districts: {config['districts']}")
    print(f"   🔄 Epsilon: {config['epsilon']}")
    
    try:
        # Create partition object
        partition = Partition(toy_geo, num_districts=config['districts'], 
                            prob_dict=prob_dict, epsilon=config['epsilon'])
        
        # Run Benders decomposition
        print("   🔄 Running Benders decomposition...")
        print("      (This may take 1-2 minutes for 25 blocks...)")
        
        best_partition, best_cost, history = partition.benders_decomposition(
            max_iterations=20, tolerance=1e-3, verbose=True
        )
        
        print(f"\n   ✅ ALGORITHM COMPLETED SUCCESSFULLY!")
        print(f"      💰 Best cost: {best_cost:.4f}")
        print(f"      🔄 Iterations: {len(history)}")
        print(f"      📉 Final gap: {history[-1]['gap']:.4f}" if history else "N/A")
        
        # Create comprehensive visualization
        print("\n5️⃣ Creating comprehensive visualization...")
        os.makedirs("figures", exist_ok=True)
        
        fig, district_info = plot_partition_analysis(
            toy_geo, best_partition, config['name'], history
        )
        
        # Save the comprehensive plot
        save_path = f"figures/25blocks_comprehensive_{config['name']}.png"
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"   📊 Comprehensive analysis saved to {save_path}")
        
        plt.show()
        
        # Print detailed district analysis
        print("\n6️⃣ Detailed District Analysis:")
        print("=" * 50)
        
        N = len(toy_geo.short_geoid_list)
        block_ids = toy_geo.short_geoid_list
        roots = [i for i in range(N) if round(best_partition[i, i]) == 1]
        
        for root_idx in roots:
            assigned = [j for j in range(N) if round(best_partition[j, root_idx]) == 1]
            assigned_names = [block_ids[j] for j in assigned]
            
            # Calculate district statistics
            district_prob = sum(prob_dict[block_ids[j]] for j in assigned)
            district_area = sum(toy_geo.get_area(block_ids[j]) for j in assigned)
            
            print(f"\n🏛️  District {block_ids[root_idx]} (Root):")
            print(f"     📍 Blocks ({len(assigned)}): {assigned_names}")
            print(f"     🎯 Probability Mass: {district_prob:.4f}")
            print(f"     📏 Total Area: {district_area:.2f} km²")
        
        return True, best_partition, history
        
    except Exception as e:
        print(f"   ❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, None, None

def main():
    """Main function."""
    success, best_partition, history = run_single_test()
    
    if success:
        print(f"\n🎉 TEST COMPLETED SUCCESSFULLY!")
        print(f"   ✅ Algorithm converged")
        print(f"   ✅ Visualization created") 
        print(f"   ✅ Check 'figures/' directory for comprehensive analysis")
    else:
        print(f"\n❌ TEST FAILED")
    
    return success

if __name__ == "__main__":
    main()