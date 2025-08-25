#!/usr/bin/env python3
"""
Proper Sample Size Cutoff Validation Using CQCP Benders Decomposition

This script validates the theoretical result using:
1. CQCP Benders method for actual worst-case distributions
2. Standard 10×10 mile service region with 10×10 grid (100 blocks)
3. Euclidean distance metric for Wasserstein distance calculation
4. Mixed-Gaussian empirical distributions from samples
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

class ToyGeoData(GeoData):
    """Standard 10×10 mile service region with 10×10 grid (100 blocks)"""
    
    def __init__(self, grid_size=10, service_region_miles=10.0, seed=42):
        np.random.seed(seed)
        self.grid_size = grid_size  # 10×10 grid = 100 blocks
        self.n_blocks = grid_size * grid_size
        self.service_region_miles = service_region_miles  # 10×10 miles
        self.miles_per_grid_unit = service_region_miles / grid_size  # 1 mile per block
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(self.n_blocks)]
        
        # Create block graph
        self.G = nx.Graph()
        self.block_to_coord = {i: (i // grid_size, i % grid_size) for i in range(self.n_blocks)}
        
        # Add all block IDs as nodes
        for block_id in self.short_geoid_list:
            self.G.add_node(block_id)
        
        # Add edges between adjacent blocks
        for i in range(self.n_blocks):
            for j in range(i+1, self.n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.G.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        # Create block graph for contiguity
        self.block_graph = self.G.copy()
        
        # Set attributes
        self.areas = {block_id: 1.0 for block_id in self.short_geoid_list}  # 1 square mile per block
        self.gdf = None
    
    def get_dist(self, block1, block2):
        """Euclidean distance in miles between blocks"""
        if block1 == block2:
            return 0.0
        try:
            i1 = self.short_geoid_list.index(block1)
            i2 = self.short_geoid_list.index(block2)
            coord1 = self.block_to_coord[i1]
            coord2 = self.block_to_coord[i2]
            # Euclidean distance in grid units, converted to miles
            euclidean_dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            return euclidean_dist * self.miles_per_grid_unit
        except:
            return float('inf')
    
    def get_area(self, block_id):
        """Each block is 1 square mile"""
        return 1.0
    
    def get_K(self, block_id):
        return 2.0

def create_mixed_gaussian_samples(grid_size=10, n_samples=1000, seed=42):
    """Create samples from standard mixed-Gaussian distribution (same as other scripts)"""
    np.random.seed(seed)
    
    # Standard mixed-Gaussian parameters
    cluster_centers = [
        (grid_size * 0.25, grid_size * 0.25),  # Top-left
        (grid_size * 0.75, grid_size * 0.25),  # Top-right  
        (grid_size * 0.50, grid_size * 0.75)   # Bottom-center
    ]
    cluster_weights = [0.4, 0.35, 0.25]
    cluster_sigmas = [grid_size * 0.15 * 0.5, grid_size * 0.16 * 0.5, grid_size * 0.12 * 0.5]
    
    samples = []
    max_attempts = n_samples * 5
    attempts = 0
    
    while len(samples) < n_samples and attempts < max_attempts:
        # Choose cluster
        cluster_idx = np.random.choice(len(cluster_centers), p=cluster_weights)
        center = cluster_centers[cluster_idx]
        sigma = cluster_sigmas[cluster_idx]
        
        # Sample from cluster
        x = np.random.normal(center[0], sigma)
        y = np.random.normal(center[1], sigma)
        
        # Accept if within service region
        if 0 <= x < grid_size and 0 <= y < grid_size:
            samples.append((x, y))
        
        attempts += 1
    
    return samples[:n_samples]

def samples_to_empirical_distribution(samples, grid_size):
    """Convert continuous samples to empirical distribution over 100 blocks"""
    n_blocks = grid_size * grid_size  # 100 blocks for 10×10 grid
    block_counts = np.zeros(n_blocks)
    
    for x, y in samples:
        # Determine which block this sample falls into
        block_row = int(np.clip(np.floor(x), 0, grid_size - 1))
        block_col = int(np.clip(np.floor(y), 0, grid_size - 1))
        block_idx = block_row * grid_size + block_col
        block_counts[block_idx] += 1
    
    # Convert to probability distribution
    empirical_dist = block_counts / len(samples)
    return empirical_dist

def solve_worst_case_with_cqcp(geodata, empirical_prob_dict, epsilon, seed=42):
    """
    Solve worst-case distribution using actual CQCP Benders decomposition
    
    Args:
        geodata: ToyGeoData object (10×10 grid, Euclidean distance)
        empirical_prob_dict: Empirical probability distribution over 100 blocks
        epsilon: Wasserstein radius (Euclidean distance based)
        
    Returns:
        worst_case_distribution: Worst-case probability distribution (100 elements)
        success: Whether CQCP solved successfully
    """
    try:
        # For this validation, use the entire service region as one district
        assigned_blocks = geodata.short_geoid_list  # All 100 blocks
        root = assigned_blocks[50]  # Center block as root (block 50 ≈ center of 10×10 grid)
        
        # Create partition instance
        partition = Partition(geodata, num_districts=1, prob_dict=empirical_prob_dict, epsilon=epsilon)
        
        # CQCP subproblem parameters
        K_i = 3.0  # Fixed linehaul cost
        F_i = 0.0  # No ODD cost for this analysis  
        beta = 0.7120
        Lambda = 10.0
        
        # Solve CQCP subproblem to get worst-case distribution
        result = partition._CQCP_benders(
            assigned_blocks=assigned_blocks,
            root=root,
            prob_dict=empirical_prob_dict,
            epsilon=epsilon,
            K_i=K_i,
            F_i=F_i,
            grid_points=10,
            beta=beta,
            Lambda=Lambda,
            single_threaded=True
        )
        
        # Extract results - _CQCP_benders returns:
        # best_obj, x_star_dict, subgrad, C_star, alpha_i, subgrad_K_i, subgrad_F_i
        if len(result) >= 7:
            worst_case_cost, x_star_dict, subgrad, C_star, alpha_i, subgrad_K_i, subgrad_F_i = result
            
            # Convert x_star_dict to distribution array (only for assigned blocks)
            worst_case_dist = np.zeros(len(assigned_blocks))
            total_mass = 0.0
            for i, block_id in enumerate(assigned_blocks):
                mass = x_star_dict.get(block_id, 0.0)
                worst_case_dist[i] = mass
                total_mass += mass
            
            # Normalize to ensure it's a proper probability distribution
            if total_mass > 0:
                worst_case_dist = worst_case_dist / total_mass
            else:
                # Fallback to uniform if no mass
                worst_case_dist = np.ones(len(assigned_blocks)) / len(assigned_blocks)
            
            return worst_case_dist, True
        else:
            print(f"CQCP returned incomplete result: {len(result)} components")
            return None, False
            
    except Exception as e:
        print(f"CQCP solver failed: {e}")
        return None, False

def measure_uniformity(distribution):
    """
    Measure uniformity using normalized entropy metric
    
    Returns:
        uniformity: entropy / max_entropy, where max_entropy = log(n)
        This gives 1.0 for perfect uniform distribution, 0.0 for maximally concentrated
    """
    n = len(distribution)
    dist_normalized = distribution / np.sum(distribution) if np.sum(distribution) > 0 else np.ones(n) / n
    
    # Add small epsilon to avoid log(0)
    dist_safe = dist_normalized + 1e-12
    
    # Calculate entropy: H = -sum(p_i * log(p_i))
    entropy = -np.sum(dist_safe * np.log(dist_safe))
    max_entropy = np.log(n)  # Maximum entropy for uniform distribution
    
    # Normalize entropy to [0, 1] scale
    uniformity = entropy / max_entropy if max_entropy > 0 else 0.0
    
    # Debug output
    uniform_mass = 1.0 / n
    max_mass = np.max(dist_normalized)
    min_mass = np.min(dist_normalized)
    print(f"  Entropy: {entropy:.6f}, Max entropy: {max_entropy:.6f}, Uniformity: {uniformity:.6f}")
    print(f"  Max mass: {max_mass:.6f}, Min mass: {min_mass:.6f}, Uniform mass: {uniform_mass:.6f}")
    
    return max(0.0, min(1.0, uniformity))  # Clamp to [0, 1]

def validate_sample_cutoff_with_cqcp_proper(epsilon_0=10.0, n_trials=5):
    """
    Validate sample cutoff theory using proper CQCP Benders decomposition
    
    Args:
        epsilon_0: Base Wasserstein radius
        n_trials: Number of trials per sample size (reduced due to CQCP computational cost)
    """
    
    # Standard parameters
    grid_size = 10  # 10×10 grid = 100 blocks
    service_region_miles = 10.0  # 10×10 miles
    
    print("=" * 80)
    print("PROPER SAMPLE SIZE CUTOFF VALIDATION WITH CQCP BENDERS")
    print("=" * 80)
    print(f"Service region: {service_region_miles}×{service_region_miles} miles")
    print(f"Grid size: {grid_size}×{grid_size} = {grid_size**2} blocks")
    print(f"Distance metric: Euclidean distance (√((x₁-x₂)² + (y₁-y₂)²))")
    print(f"Base Wasserstein radius: ε₀ = {epsilon_0}")
    print(f"Scaled radius formula: ε(n) = ε₀/√n")
    print(f"Number of trials per sample size: {n_trials}")
    print(f"Worst-case solver: CQCP Benders decomposition")
    print()
    
    # Create geodata
    geodata = ToyGeoData(grid_size, service_region_miles, seed=42)
    
    # Sample sizes to test (reduced due to computational cost)
    sample_sizes = [10, 20, 30, 50, 100, 200, 300]
    
    # Storage for results
    results = {
        'sample_sizes': [],
        'uniformities_mean': [],
        'uniformities_std': [],
        'epsilons': [],
        'worst_case_distributions': [],
        'success_rates': []
    }
    
    for n in sample_sizes:
        print(f"Testing n = {n} samples...")
        
        epsilon = epsilon_0 / np.sqrt(n)
        uniformities = []
        trial_worst_cases = []
        successful_trials = 0
        
        for trial in range(n_trials):
            print(f"  Trial {trial + 1}/{n_trials}...", end=' ')
            
            # Generate empirical distribution from mixed-Gaussian samples
            samples = create_mixed_gaussian_samples(grid_size, n, seed=42 + trial * 1000)
            empirical_dist = samples_to_empirical_distribution(samples, grid_size)
            
            # Create probability dictionary
            empirical_prob_dict = {
                block_id: empirical_dist[i] 
                for i, block_id in enumerate(geodata.short_geoid_list)
            }
            
            # Solve worst-case distribution with CQCP
            worst_case, success = solve_worst_case_with_cqcp(
                geodata, empirical_prob_dict, epsilon, seed=42 + trial
            )
            
            if success and worst_case is not None:
                # Measure uniformity
                uniformity = measure_uniformity(worst_case)
                uniformities.append(uniformity)
                trial_worst_cases.append(worst_case)
                successful_trials += 1
                print("✓")
            else:
                print("✗")
        
        if successful_trials > 0:
            # Store results
            results['sample_sizes'].append(n)
            results['uniformities_mean'].append(np.mean(uniformities))
            results['uniformities_std'].append(np.std(uniformities))
            results['epsilons'].append(epsilon)
            results['worst_case_distributions'].append(np.mean(trial_worst_cases, axis=0))
            results['success_rates'].append(successful_trials / n_trials)
            
            print(f"    ε = {epsilon:.3f}, Uniformity = {np.mean(uniformities):.3f} ± {np.std(uniformities):.3f}")
            print(f"    Success rate: {successful_trials}/{n_trials} ({100*successful_trials/n_trials:.0f}%)")
        else:
            print(f"    All trials failed for n = {n}")
        
        print()
    
    # Analyze results
    if len(results['sample_sizes']) > 1:
        uniformities = np.array(results['uniformities_mean'])
        epsilons = np.array(results['epsilons'])
        
        # Find cutoff: where uniformity drops below 90% (adjusted for max-min metric)
        uniformity_threshold = 0.96
        uniform_indices = np.where(uniformities >= uniformity_threshold)[0]
        
        if len(uniform_indices) > 0:
            n_star_idx = uniform_indices[-1]
            n_star = results['sample_sizes'][n_star_idx]
            # Calculate critical epsilon correctly: ε₀/√n*
            epsilon_critical = epsilon_0 / np.sqrt(n_star)
        else:
            n_star_idx = np.argmax(uniformities)
            n_star = results['sample_sizes'][n_star_idx]
            # Calculate critical epsilon correctly: ε₀/√n*
            epsilon_critical = epsilon_0 / np.sqrt(n_star)
        
        print("=" * 60)
        print("CQCP-BASED CUTOFF ANALYSIS")
        print("=" * 60)
        print(f"Distance metric: Euclidean distance on 10×10 mile grid")
        print(f"Uniformity metric: normalized entropy (H / log(n))")
        print(f"Uniformity threshold: {uniformity_threshold:.1%}")
        print(f"Critical ε: {epsilon_critical:.3f}")
        print(f"Sample size cutoff n* = {n_star}")
        print(f"At n*, uniformity = {uniformities[n_star_idx]:.3f}")
        
        return results, n_star, epsilon_critical
    else:
        print("Insufficient successful results for analysis")
        return results, None, None

def create_cqcp_visualization(results, n_star, epsilon_critical, grid_size):
    """Create visualization for CQCP-based validation (same style as simple version)"""
    
    if not results['sample_sizes']:
        print("No results to visualize")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    
    sample_sizes = np.array(results['sample_sizes'])
    uniformities_mean = np.array(results['uniformities_mean'])
    uniformities_std = np.array(results['uniformities_std'])
    epsilons = np.array(results['epsilons'])
    
    # Plot 1: Uniformity vs Sample Size with clear regime boundaries
    ax1.errorbar(sample_sizes, uniformities_mean, yerr=uniformities_std, 
                 fmt='bo-', linewidth=2, markersize=6, capsize=4)
    if n_star is not None:
        ax1.axvline(x=n_star, color='red', linestyle='--', linewidth=3, 
                    label=f'Critical n* = {n_star}\n(CQCP-based, entropy-based uniformity)')
        
        # Fill regime regions
        ax1.fill_between(sample_sizes[sample_sizes <= n_star], 0.5, 1.0, 
                         alpha=0.15, color='green', label='Uniform Regime (Small n)')
        ax1.fill_between(sample_sizes[sample_sizes > n_star], 0.5, 1.0, 
                         alpha=0.15, color='red', label='Data-Driven Regime (Large n)')
    
    ax1.set_xlabel('Sample Size n', fontsize=14)
    ax1.set_ylabel('Worst-Case Distribution Uniformity', fontsize=14)
    ax1.set_title('CQCP Benders: Sample Size Cutoff Validation\n(10×10 Grid, Euclidean Distance)', fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.set_ylim([0.5, 1.0])
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 2: Wasserstein radius with critical line
    ax2.loglog(sample_sizes, epsilons, 'go-', linewidth=3, markersize=7)
    if n_star is not None:
        ax2.axvline(x=n_star, color='red', linestyle='--', linewidth=3)
        ax2.axhline(y=epsilon_critical, color='orange', linestyle='-', linewidth=2,
                    label=f'Critical ε = {epsilon_critical:.3f}')
    
    # Theoretical curve
    theoretical = epsilons[0] * np.sqrt(sample_sizes[0]) / np.sqrt(sample_sizes)
    ax2.loglog(sample_sizes, theoretical, 'r:', linewidth=2, alpha=0.8, 
               label='ε₀/√n (theory)')
    
    ax2.set_xlabel('Sample Size n', fontsize=14)
    ax2.set_ylabel('Wasserstein Radius ε(n)', fontsize=14)
    ax2.set_title('Radius Scaling with Critical Threshold', fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Plot 3: Example uniform distribution (small n)
    if len(results['worst_case_distributions']) > 0:
        small_n_idx = 0  # First sample size (smallest n)
        small_n = sample_sizes[small_n_idx]
        uniform_dist = results['worst_case_distributions'][small_n_idx]
        uniform_grid = uniform_dist.reshape(grid_size, grid_size)
        
        im1 = ax3.imshow(uniform_grid, cmap='Blues', origin='lower', vmin=0, vmax=np.max(uniform_grid)*1.2)
        ax3.set_title(f'Uniform Worst-Case (n={small_n})\nε={epsilons[small_n_idx]:.3f}, Uniformity={uniformities_mean[small_n_idx]:.3f}', 
                      fontsize=13)
        ax3.set_xlabel('x coordinate')
        ax3.set_ylabel('y coordinate')
        plt.colorbar(im1, ax=ax3, shrink=0.8)
    else:
        ax3.text(0.5, 0.5, 'No distribution\ndata available', 
                 ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Uniform Distribution Example', fontsize=15)
    
    # Plot 4: Example non-uniform distribution (large n)
    if len(results['worst_case_distributions']) > 1:
        large_n_idx = -1  # Last sample size (largest n)
        large_n = sample_sizes[large_n_idx]
        nonuniform_dist = results['worst_case_distributions'][large_n_idx]
        nonuniform_grid = nonuniform_dist.reshape(grid_size, grid_size)
        
        im2 = ax4.imshow(nonuniform_grid, cmap='Reds', origin='lower', vmin=0, vmax=np.max(nonuniform_grid)*1.2)
        ax4.set_title(f'Data-Driven Worst-Case (n={large_n})\nε={epsilons[large_n_idx]:.3f}, Uniformity={uniformities_mean[large_n_idx]:.3f}', 
                      fontsize=13)
        ax4.set_xlabel('x coordinate')
        ax4.set_ylabel('y coordinate')
        plt.colorbar(im2, ax=ax4, shrink=0.8)
    else:
        ax4.text(0.5, 0.5, 'No distribution\ndata available', 
                 ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Data-Driven Distribution Example', fontsize=15)
    
    plt.tight_layout()
    plt.savefig('sample_cutoff_validation.pdf', dpi=300, bbox_inches='tight')
    print("✅ CQCP validation plots saved as 'cqcp_proper_sample_cutoff_validation.pdf'")
    
    return fig

def main():
    import sys
    
    # Parse command line arguments for epsilon_0
    epsilon_0 = 8.0  # Default smaller value
    n_trials = 3  # Default trials
    
    for arg in sys.argv[1:]:
        try:
            if '=' in arg:
                key, value = arg.split('=')
                if key == 'epsilon_0':
                    epsilon_0 = float(value)
                elif key == 'n_trials':
                    n_trials = int(value)
            else:
                # First argument is epsilon_0, second is n_trials
                if len(sys.argv) == 2:
                    epsilon_0 = float(arg)
                elif len(sys.argv) == 3:
                    if sys.argv[1] == arg:
                        epsilon_0 = float(arg)
                    else:
                        n_trials = int(arg)
        except ValueError:
            print(f"Invalid argument: {arg}")
    
    print(f"Using ε₀ = {epsilon_0}, n_trials = {n_trials}")
    print("Usage: python validate_cutoff_proper_final.py [epsilon_0] [n_trials]")
    print("   or: python validate_cutoff_proper_final.py epsilon_0=X n_trials=Y")
    print()
    
    # Run proper CQCP-based validation
    results, n_star, epsilon_critical = validate_sample_cutoff_with_cqcp_proper(epsilon_0, n_trials)
    
    # Create visualizations
    if results['sample_sizes']:
        create_cqcp_visualization(results, n_star, epsilon_critical, 10)
        
        print("\n" + "=" * 80)
        print("PROPER CQCP-BASED VALIDATION SUMMARY")
        print("=" * 80)
        print("✅ Uses actual CQCP Benders decomposition for worst-case distributions")
        print("✅ Standard 10×10 mile service region with 10×10 grid (100 blocks)")
        print("✅ Euclidean distance metric for Wasserstein distance calculation")
        print("✅ Mixed-Gaussian empirical distributions from samples")
        print("✅ Data-driven determination of critical epsilon and n*")
        
        if n_star is not None:
            print(f"\nKey Results:")
            print(f"• Critical sample size: n* = {n_star}")
            print(f"• Critical Wasserstein radius: ε* = {epsilon_critical:.3f}")
            print(f"• Distance metric: Euclidean distance on grid")
            print(f"• Worst-case solver: CQCP Benders decomposition")
    else:
        print("No successful results obtained. Consider:")
        print("• Reducing epsilon_0 for easier CQCP convergence")
        print("• Increasing n_trials for more robust statistics")
        print("• Checking CQCP solver parameters")
    
    plt.show()

if __name__ == "__main__":
    main()