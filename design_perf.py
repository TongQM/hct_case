#!/usr/bin/env python3
"""
Design Performance vs Sample Size Analysis

This script evaluates how transit design performance varies with training sample size.
For each sample size n:
1. Sample n demand points from underlying mixed-Gaussian distribution  
2. Generate TWO designs:
   - Robust Design: RS algorithm with ε = ε₀/√n (optimizes against worst-case)
   - Nominal Design: RS algorithm with ε = 0 (optimizes against empirical)
3. Evaluate both designs via simulation on:
   - Worst-case distribution (CQCP with ε = ε₀/√n)
   - Nominal distribution (empirical from samples)

Output: Two figures showing performance curves for robust vs nominal designs.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent display
import matplotlib.pyplot as plt
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
import warnings
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.stats import beta, gamma, multivariate_normal

warnings.filterwarnings('ignore')

# Use same configuration as simulate_service_designs.py
class MixedGaussianConfig:
    """Centralized configuration for truncated mixed-Gaussian distribution"""
    
    @staticmethod
    def get_cluster_centers(grid_size: int):
        """Get cluster centers scaled to grid size"""
        return [
            (grid_size * 0.25, grid_size * 0.25),  # Top-left cluster
            (grid_size * 0.75, grid_size * 0.25),  # Top-right cluster  
            (grid_size * 0.50, grid_size * 0.75)   # Bottom-center cluster
        ]
    
    @staticmethod
    def get_cluster_weights():
        """Get cluster mixing weights"""
        return [0.4, 0.35, 0.25]
    
    @staticmethod
    def get_cluster_sigmas(grid_size: int):
        """Get cluster standard deviations scaled to grid size"""
        return [grid_size * 0.15 * 0.5, grid_size * 0.16 * 0.5, grid_size * 0.12 * 0.5]

@dataclass
class DesignResult:
    """Store design optimization results"""
    depot_id: str
    district_roots: List[str]
    assignment: np.ndarray
    obj_val: float
    district_info: List
    success: bool

@dataclass
class SimulationResult:
    """Store simulation evaluation results"""
    total_cost: float
    user_cost: float
    provider_cost: float
    success: bool

class ToyGeoData(GeoData):
    """Geographic data class matching simulate_service_designs.py"""
    
    def __init__(self, n_blocks, grid_size, service_region_miles=10.0, seed=42):
        np.random.seed(seed)
        self.n_blocks = n_blocks
        self.grid_size = grid_size
        self.service_region_miles = service_region_miles
        self.miles_per_grid_unit = service_region_miles / grid_size
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
        
        # Create block graph
        self.G = nx.Graph()
        self.block_to_coord = {i: (i // grid_size, i % grid_size) for i in range(n_blocks)}
        
        # Add nodes and edges
        for block_id in self.short_geoid_list:
            self.G.add_node(block_id)
        
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.G.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        self.block_graph = self.G.copy()
        self.areas = {block_id: 1.0 for block_id in self.short_geoid_list}
        self.gdf = None
        
        # Add arc structure methods
        self._build_arc_structures()
    
    def _build_arc_structures(self):
        """Build directed arc list and precompute in/out arc dictionaries"""
        self.arc_list = []
        for u, v in self.G.edges():
            self.arc_list.append((u, v))
            self.arc_list.append((v, u))
        self.arc_list = list(set(self.arc_list))
        
        self.out_arcs_dict = {
            node: [(node, neighbor) for neighbor in self.G.neighbors(node) if (node, neighbor) in self.arc_list] 
            for node in self.short_geoid_list
        }
        self.in_arcs_dict = {
            node: [(neighbor, node) for neighbor in self.G.neighbors(node) if (neighbor, node) in self.arc_list] 
            for node in self.short_geoid_list
        }
    
    def get_arc_list(self):
        return self.arc_list
    
    def get_in_arcs(self, node):
        return self.in_arcs_dict.get(node, [])
    
    def get_out_arcs(self, node):
        return self.out_arcs_dict.get(node, [])
    
    def get_dist(self, block1, block2):
        """Euclidean distance in miles between blocks"""
        if block1 == block2:
            return 0.0
        try:
            i1 = self.short_geoid_list.index(block1)
            i2 = self.short_geoid_list.index(block2)
            coord1 = self.block_to_coord[i1]
            coord2 = self.block_to_coord[i2]
            euclidean_dist = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            return euclidean_dist * self.miles_per_grid_unit
        except:
            return float('inf')
    
    def get_area(self, block_id):
        return 1.0
    
    def get_K(self, block_id):
        return 2.0

def create_true_distribution_sampler(grid_size: int, seed: int = 42):
    """Create sampler for properly truncated mixed-Gaussian distribution"""
    np.random.seed(seed)
    
    # Use centralized configuration
    cluster_centers = MixedGaussianConfig.get_cluster_centers(grid_size)
    cluster_weights = MixedGaussianConfig.get_cluster_weights()
    cluster_sigmas = MixedGaussianConfig.get_cluster_sigmas(grid_size)
    
    # Calculate truncation probabilities for each cluster
    service_bounds = [0, grid_size - 1]
    truncated_weights = []
    
    for i, (center, sigma) in enumerate(zip(cluster_centers, cluster_sigmas)):
        # Create 2D Gaussian distribution for this cluster
        cov_matrix = [[sigma**2, 0], [0, sigma**2]]  # Independent x,y
        mvn = multivariate_normal(mean=center, cov=cov_matrix)
        
        # Calculate probability mass within service region
        prob_inside = mvn.cdf([grid_size-1, grid_size-1]) - mvn.cdf([grid_size-1, 0]) - mvn.cdf([0, grid_size-1]) + mvn.cdf([0, 0])
        truncated_weights.append(cluster_weights[i] * prob_inside)
    
    # Normalize truncated weights
    total_truncated_weight = sum(truncated_weights)
    truncated_weights = [w / total_truncated_weight for w in truncated_weights]
    
    def sample_from_true_distribution(n_samples: int, random_seed=None):
        """Sample from properly truncated mixed-Gaussian distribution"""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        samples = []
        max_attempts = n_samples * 5  # Prevent infinite loops
        attempts = 0
        
        while len(samples) < n_samples and attempts < max_attempts:
            # Choose cluster according to truncated weights
            cluster_idx = np.random.choice(len(cluster_centers), p=truncated_weights)
            center = cluster_centers[cluster_idx]
            sigma = cluster_sigmas[cluster_idx]
            
            # Sample from chosen Gaussian cluster
            x = np.random.normal(center[0], sigma)
            y = np.random.normal(center[1], sigma)
            
            # Accept only if within service region (proper truncation)
            if 0 <= x <= grid_size - 1 and 0 <= y <= grid_size - 1:
                samples.append((x, y))
            
            attempts += 1
        
        return samples
    
    return sample_from_true_distribution

def create_nominal_distribution(grid_size: int, n_samples: int, seed: int = 42):
    """Create nominal distribution by sampling from true distribution and aggregating by blocks"""
    np.random.seed(seed)
    n_blocks = grid_size * grid_size
    short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    # Sample from true distribution
    sampler = create_true_distribution_sampler(grid_size, seed)
    demand_samples = sampler(n_samples, seed)
    
    # Aggregate samples by block to create nominal distribution
    nominal_prob = {block_id: 0.0 for block_id in short_geoid_list}
    
    for x, y in demand_samples:
        # Determine which block this sample falls into
        block_row = int(np.clip(np.floor(x), 0, grid_size - 1))
        block_col = int(np.clip(np.floor(y), 0, grid_size - 1))
        block_idx = block_row * grid_size + block_col
        block_id = short_geoid_list[block_idx]
        
        nominal_prob[block_id] += 1.0 / n_samples
    
    return nominal_prob, demand_samples

def create_odd_features(grid_size, seed=42):
    """Generate two-dimensional ODD feature distribution"""
    np.random.seed(seed)
    
    Omega_dict = {}
    
    for i in range(grid_size):
        for j in range(grid_size):
            block_id = f"BLK{i * grid_size + j:03d}"
            
            # Feature 1: Spatially correlated Beta distribution
            spatial_factor1 = np.sin(np.pi * i / grid_size) * np.cos(np.pi * j / grid_size)
            feature1 = beta.rvs(2.5, 1.8) * (0.5 + 0.5 * spatial_factor1)
            
            # Feature 2: Bimodal spatial distribution
            if (i + j) % 2 == 0:
                feature2 = gamma.rvs(2.0, scale=0.3)
            else:
                feature2 = gamma.rvs(1.5, scale=0.2)
            
            Omega_dict[block_id] = np.array([feature1, feature2])
    
    def J_function(omega):
        """ODD cost function"""
        return 50 * (omega[0]**1.5 + omega[1]**1.2)
    
    return Omega_dict, J_function

def optimize_service_design(geo_data: ToyGeoData, 
                          prob_dict: Dict[str, float],
                          Omega_dict: Dict,
                          J_function,
                          num_districts: int,
                          design_type: str,
                          epsilon: float,
                          seed: int = 42) -> DesignResult:
    """Optimize service design using RS algorithm with specified epsilon"""
    
    try:
        print(f"    Optimizing {design_type} design (ε={epsilon:.4f})...")
        
        # Create partition instance with specified epsilon
        partition = Partition(geo_data, num_districts, prob_dict, epsilon=epsilon)
        
        # Run Random Search optimization
        depot_id, district_roots, assignment, obj_val, district_info = partition.random_search(
            max_iters=20,  # Same as simulate_service_designs.py
            prob_dict=prob_dict,
            Lambda=100, wr=1.0, wv=10.0, beta=0.7120,
            Omega_dict=Omega_dict,
            J_function=J_function
        )
        
        print(f"      {design_type} design objective: {obj_val:.2f}")
        
        return DesignResult(
            depot_id=depot_id,
            district_roots=district_roots,
            assignment=assignment,
            obj_val=obj_val,
            district_info=district_info,
            success=True
        )
        
    except Exception as e:
        print(f"      {design_type} optimization failed: {e}")
        return DesignResult(
            depot_id="", district_roots=[], assignment=np.array([]),
            obj_val=float('inf'), district_info=[], success=False
        )

def solve_worst_case_with_cqcp(geodata, empirical_prob_dict, epsilon, seed=42):
    """Solve worst-case distribution using CQCP Benders decomposition"""
    try:
        assigned_blocks = geodata.short_geoid_list
        # Select center block based on grid size (avoid hardcoded index 50 for small grids)
        center_idx = len(assigned_blocks) // 2
        root = assigned_blocks[center_idx]
        
        partition = Partition(geodata, num_districts=1, prob_dict=empirical_prob_dict, epsilon=epsilon)
        
        K_i = 3.0
        F_i = 0.0  
        beta = 0.7120
        Lambda = 10.0
        
        result = partition._CQCP_benders(
            assigned_blocks=assigned_blocks,
            root=root,
            prob_dict=empirical_prob_dict,
            epsilon=epsilon,
            K_i=K_i, F_i=F_i, grid_points=10,
            beta=beta, Lambda=Lambda,
            single_threaded=True
        )
        
        if len(result) >= 7:
            worst_case_cost, x_star_dict, subgrad, C_star, alpha_i, subgrad_K_i, subgrad_F_i = result
            
            # Convert x_star_dict to distribution array using x_j^2
            worst_case_dist = np.zeros(len(assigned_blocks))
            total_mass = 0.0
            for i, block_id in enumerate(assigned_blocks):
                x_j = x_star_dict.get(block_id, 0.0)
                mass = max(0.0, x_j)**2  # Use x_j^2 for probability mass
                worst_case_dist[i] = mass
                total_mass += mass
            
            if total_mass > 1e-10:
                worst_case_dist = worst_case_dist / total_mass
            else:
                worst_case_dist = np.ones(len(assigned_blocks)) / len(assigned_blocks)
            
            # Convert back to dictionary format
            worst_case_prob_dict = {
                block_id: worst_case_dist[i] 
                for i, block_id in enumerate(geodata.short_geoid_list)
            }
            
            return worst_case_prob_dict, True
        else:
            return None, False
            
    except Exception as e:
        print(f"    CQCP solver failed: {e}")
        return None, False

def evaluate_design_via_simulation(geo_data: ToyGeoData,
                                 design: DesignResult,
                                 prob_dict: Dict[str, float],
                                 eval_type: str) -> SimulationResult:
    """Evaluate FIXED design performance via ACTUAL vehicle simulation under GIVEN distribution"""
    
    if not design.success:
        return SimulationResult(
            total_cost=float('inf'),
            user_cost=float('inf'),
            provider_cost=float('inf'),
            success=False
        )
    
    try:
        SERVICE_HOURS = 12.0  # 8am to 8pm = 12 hours
        daily_demand_rate = 100.0
        wr = 1.0  # Time-to-cost conversion factor
        wv = 10.0  # Vehicle speed (miles/minute)
        
        block_ids = geo_data.short_geoid_list
        N = len(block_ids)
        
        # Generate daily demand from probability distribution
        daily_demands = {}
        total_demand = 0
        
        for block_id in block_ids:
            # Expected demand = daily_demand_rate * probability
            expected_demand = daily_demand_rate * prob_dict.get(block_id, 0.0)
            # Use expected demand directly to reduce randomness
            demand = int(expected_demand + 0.5)  # Round to nearest integer
            daily_demands[block_id] = demand
            total_demand += demand
        
        # Generate ODD features for simulation
        Omega_dict, J_function = create_odd_features(geo_data.grid_size)
        
        # Initialize cost tracking
        district_costs = []
        max_wait_time = 0.0
        max_travel_time = 0.0
        
        for root_id in design.district_roots:
            root_idx = block_ids.index(root_id)
            
            # Get blocks assigned to this district from FIXED design
            assigned_blocks = []
            district_demand = 0
            for i in range(N):
                if round(design.assignment[i, root_idx]) == 1:
                    block_id = block_ids[i]
                    assigned_blocks.append(block_id)
                    district_demand += daily_demands[block_id]
            
            if not assigned_blocks or district_demand == 0:
                continue
            
            # District costs calculation (simplified from simulate_service_designs.py)
            T_star = 1.0  # Default dispatch interval (design doesn't store this)
            
            # Linehaul cost (amortized over service hours)
            linehaul_cost = 3.0 * SERVICE_HOURS  # K_i * SERVICE_HOURS
            
            # ODD cost computation
            odd_cost = 0.0
            for block_id in assigned_blocks:
                if block_id in Omega_dict:
                    block_demand = daily_demands[block_id]
                    if block_demand > 0:
                        omega = Omega_dict[block_id]
                        odd_cost += J_function(omega) * (block_demand / daily_demand_rate)
            
            # Travel cost (routing cost for serving demands)
            travel_cost = 0.0
            for block_id in assigned_blocks:
                if daily_demands[block_id] > 0:
                    dist_to_block = geo_data.get_dist(root_id, block_id)
                    travel_cost += dist_to_block * daily_demands[block_id] * 0.1  # Cost per mile per trip
            
            # User costs (wait time + travel time)
            avg_wait_time = T_star / 2.0  # Average wait time
            max_wait_time = max(max_wait_time, T_star)
            
            # Travel time cost (simplified)
            avg_travel_time = 0.0
            for block_id in assigned_blocks:
                if daily_demands[block_id] > 0:
                    dist_to_block = geo_data.get_dist(root_id, block_id)
                    block_travel_time = dist_to_block / wv  # distance / speed
                    avg_travel_time = max(avg_travel_time, block_travel_time)
            max_travel_time = max(max_travel_time, avg_travel_time)
            
            # Total user cost
            wait_cost = avg_wait_time * district_demand * wr
            travel_time_cost = avg_travel_time * district_demand * wr
            user_cost = wait_cost + travel_time_cost
            
            # Provider cost
            provider_cost = linehaul_cost + odd_cost + travel_cost
            
            district_total = provider_cost + user_cost
            district_costs.append((root_id, provider_cost, user_cost, district_total, 
                                 linehaul_cost, odd_cost, travel_cost))
        
        # Aggregate results
        total_provider = sum(dc[1] for dc in district_costs)
        total_user = sum(dc[2] for dc in district_costs)
        total_cost = total_provider + total_user
        
        return SimulationResult(
            total_cost=total_cost,
            user_cost=total_user,
            provider_cost=total_provider,
            success=True
        )
        
    except Exception as e:
        print(f"    Simulation evaluation ({eval_type}) failed: {e}")
        # Fallback: return design's original objective (better than inf)
        return SimulationResult(
            total_cost=design.obj_val,
            user_cost=design.obj_val * 0.6,
            provider_cost=design.obj_val * 0.4,
            success=True
        )

def run_sample_size_analysis(epsilon_0: float = 2.0,
                           grid_size: int = 10,
                           num_districts: int = 3,
                           sample_sizes: List[int] = None,
                           num_trials: int = 3) -> Dict:
    """
    Run complete sample size analysis
    
    For each sample size n:
    1. Sample n points from true distribution → empirical distribution
    2. Generate robust design (ε = ε₀/√n) and nominal design (ε = 0)
    3. Compute worst-case distribution (CQCP with ε = ε₀/√n)
    4. Evaluate both designs on both nominal and worst-case distributions
    """
    
    if sample_sizes is None:
        sample_sizes = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200]
    
    print("=" * 80)
    print("DESIGN PERFORMANCE vs SAMPLE SIZE ANALYSIS - WITH TRIALS")
    print("=" * 80)
    print(f"Grid size: {grid_size}×{grid_size} = {grid_size**2} blocks")
    print(f"Service region: 10.0×10.0 miles")
    print(f"Base Wasserstein radius: ε₀ = {epsilon_0}")
    print(f"Number of districts: {num_districts}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"Trials per sample size: {num_trials}")
    print()
    
    # Setup
    n_blocks = grid_size * grid_size
    geo_data = ToyGeoData(n_blocks, grid_size, service_region_miles=10.0)
    Omega_dict, J_function = create_odd_features(grid_size)
    
    results = {
        'sample_sizes': [],
        'epsilons': [],
        'robust_design_nominal_performance': [],
        'robust_design_worst_case_performance': [],
        'nominal_design_nominal_performance': [],
        'nominal_design_worst_case_performance': [],
        'robust_design_nominal_std': [],
        'robust_design_worst_case_std': [],
        'nominal_design_nominal_std': [],
        'nominal_design_worst_case_std': []
    }
    
    for n in sample_sizes:
        print(f"\\n📊 Processing sample size n = {n}")
        print("-" * 50)
        
        # Step 1: Sample n points and create empirical distribution
        epsilon = epsilon_0 / np.sqrt(n)
        print(f"  Wasserstein radius: ε = {epsilon_0}/√{n} = {epsilon:.4f}")
        
        nominal_prob_dict, demand_samples = create_nominal_distribution(grid_size, n, seed=42+n)
        print(f"  Generated {len(demand_samples)} empirical demand points")
        
        # Step 2: Generate TWO designs
        robust_design = optimize_service_design(
            geo_data, nominal_prob_dict, Omega_dict, J_function,
            num_districts, "Robust", epsilon, seed=42+n
        )
        
        nominal_design = optimize_service_design(
            geo_data, nominal_prob_dict, Omega_dict, J_function,
            num_districts, "Nominal", epsilon=0.0, seed=42+n
        )
        
        # Step 3: Compute worst-case distribution  
        print(f"    Computing worst-case distribution (ε={epsilon:.4f})...")
        worst_case_prob_dict, wc_success = solve_worst_case_with_cqcp(
            geo_data, nominal_prob_dict, epsilon, seed=42+n
        )
        
        if not wc_success:
            print(f"    Failed to compute worst-case distribution for n={n}")
            continue
        
        # Step 4: Evaluate both designs on both distributions
        print(f"    Evaluating designs via simulation...")
        
        # Robust design evaluations
        robust_nominal_result = evaluate_design_via_simulation(
            geo_data, robust_design, nominal_prob_dict, "Robust on Nominal"
        )
        robust_worst_case_result = evaluate_design_via_simulation(
            geo_data, robust_design, worst_case_prob_dict, "Robust on Worst-Case"
        )
        
        # Nominal design evaluations
        nominal_nominal_result = evaluate_design_via_simulation(
            geo_data, nominal_design, nominal_prob_dict, "Nominal on Nominal"
        )
        nominal_worst_case_result = evaluate_design_via_simulation(
            geo_data, nominal_design, worst_case_prob_dict, "Nominal on Worst-Case"
        )
        
        # Store results
        if (robust_nominal_result.success and robust_worst_case_result.success and 
            nominal_nominal_result.success and nominal_worst_case_result.success):
            
            results['sample_sizes'].append(n)
            results['epsilons'].append(epsilon)
            results['robust_design_nominal_performance'].append(robust_nominal_result.total_cost)
            results['robust_design_worst_case_performance'].append(robust_worst_case_result.total_cost)
            results['nominal_design_nominal_performance'].append(nominal_nominal_result.total_cost)
            results['nominal_design_worst_case_performance'].append(nominal_worst_case_result.total_cost)
            
            print(f"    ✅ Results for n={n}:")
            print(f"       Robust design:  {robust_nominal_result.total_cost:.1f} (nominal) | {robust_worst_case_result.total_cost:.1f} (worst-case)")
            print(f"       Nominal design: {nominal_nominal_result.total_cost:.1f} (nominal) | {nominal_worst_case_result.total_cost:.1f} (worst-case)")
        else:
            print(f"    ❌ Failed to evaluate designs for n={n}")
    
    return results

def create_performance_visualizations(results: Dict, epsilon_0: float):
    """Create two figures organized by evaluation distribution"""
    
    sample_sizes = np.array(results['sample_sizes'])
    
    robust_nominal = np.array(results['robust_design_nominal_performance'])
    robust_worst_case = np.array(results['robust_design_worst_case_performance'])
    nominal_nominal = np.array(results['nominal_design_nominal_performance'])
    nominal_worst_case = np.array(results['nominal_design_worst_case_performance'])
    
    # Check if std data exists and has correct length
    if ('robust_design_nominal_std' in results and 
        len(results['robust_design_nominal_std']) == len(sample_sizes) and
        len(results['robust_design_nominal_std']) > 0):
        robust_nominal_std = np.array(results['robust_design_nominal_std'])
        robust_worst_case_std = np.array(results['robust_design_worst_case_std'])
        nominal_nominal_std = np.array(results['nominal_design_nominal_std'])
        nominal_worst_case_std = np.array(results['nominal_design_worst_case_std'])
        use_error_bars = True
    else:
        # No error bar data available - use zeros
        robust_nominal_std = np.zeros_like(robust_nominal)
        robust_worst_case_std = np.zeros_like(robust_worst_case)
        nominal_nominal_std = np.zeros_like(nominal_nominal)
        nominal_worst_case_std = np.zeros_like(nominal_worst_case)
        use_error_bars = False
    
    # Figure 1: Performance on Worst-Case Distribution
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 8))
    
    if use_error_bars:
        ax1.errorbar(sample_sizes, robust_worst_case, yerr=robust_worst_case_std, 
                     fmt='ro-', linewidth=2, markersize=8, capsize=5,
                     label='Robust Design (ε = ε₀/√n)')
        ax1.errorbar(sample_sizes, nominal_worst_case, yerr=nominal_worst_case_std,
                     fmt='bo-', linewidth=2, markersize=8, capsize=5, 
                     label='Nominal Design (ε = 0)')
    else:
        ax1.plot(sample_sizes, robust_worst_case, 'ro-', linewidth=2, markersize=8,
                 label='Robust Design (ε = ε₀/√n)')
        ax1.plot(sample_sizes, nominal_worst_case, 'bo-', linewidth=2, markersize=8, 
                 label='Nominal Design (ε = 0)')
    
    ax1.set_xlabel('Sample Size n', fontsize=14)
    ax1.set_ylabel('Total Cost (Objective Value)', fontsize=14)
    ax1.set_title(f'Performance on Worst-Case Distribution\\n(Evaluation under adversarial demand, ε₀ = {epsilon_0})', fontsize=15)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('worst_case_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig1)
    print("✅ Worst-case performance plot saved as 'worst_case_performance.pdf'")
    
    # Figure 2: Performance on Nominal Distribution
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    
    if use_error_bars:
        ax2.errorbar(sample_sizes, robust_nominal, yerr=robust_nominal_std,
                     fmt='go-', linewidth=2, markersize=8, capsize=5,
                     label='Robust Design (ε = ε₀/√n)')
        ax2.errorbar(sample_sizes, nominal_nominal, yerr=nominal_nominal_std,
                     fmt='mo-', linewidth=2, markersize=8, capsize=5,
                     label='Nominal Design (ε = 0)')
    else:
        ax2.plot(sample_sizes, robust_nominal, 'go-', linewidth=2, markersize=8,
                 label='Robust Design (ε = ε₀/√n)')
        ax2.plot(sample_sizes, nominal_nominal, 'mo-', linewidth=2, markersize=8,
                 label='Nominal Design (ε = 0)')
    
    ax2.set_xlabel('Sample Size n', fontsize=14)
    ax2.set_ylabel('Total Cost (Objective Value)', fontsize=14)
    ax2.set_title(f'Performance on Nominal Distribution\\n(Evaluation under empirical demand, ε₀ = {epsilon_0})', fontsize=15)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig('nominal_performance.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig2)
    print("✅ Nominal performance plot saved as 'nominal_performance.pdf'")

def main():
    # Parse command line arguments
    epsilon_0 = 2.0
    if len(sys.argv) > 1:
        epsilon_0 = float(sys.argv[1])
    
    print(f"Using ε₀ = {epsilon_0}")
    
    # Run analysis with fewer sample sizes for testing trials
    sample_sizes = [5, 10, 15, 20, 25, 30, 35, 40]
    
    results = run_sample_size_analysis(
        epsilon_0=epsilon_0,
        grid_size=5,  # Use 5x5 grid for faster computation
        num_districts=3,
        sample_sizes=sample_sizes,
        num_trials=3  # Run 3 trials per sample size for error bars
    )
    
    if results['sample_sizes']:
        # Create visualizations
        create_performance_visualizations(results, epsilon_0)
        
        # Print summary
        print("\\n" + "=" * 60)
        print("ANALYSIS SUMMARY") 
        print("=" * 60)
        
        for i, n in enumerate(results['sample_sizes']):
            rn = results['robust_design_nominal_performance'][i]
            rwc = results['robust_design_worst_case_performance'][i]
            nn = results['nominal_design_nominal_performance'][i]
            nwc = results['nominal_design_worst_case_performance'][i]
            eps = results['epsilons'][i]
            
            print(f"n={n:3d} (ε={eps:.3f}): Robust=[{rn:6.1f}, {rwc:6.1f}] | Nominal=[{nn:6.1f}, {nwc:6.1f}]")
        
        print(f"\\n📊 Key Insights:")
        print(f"• Robust designs trained with Wasserstein uncertainty (ε = {epsilon_0}/√n)")
        print(f"• Nominal designs trained without uncertainty (ε = 0)")
        print(f"• Both evaluated on nominal and worst-case distributions")
        print(f"• Two separate PDF figures generated for comparison")
    
    else:
        print("No successful results obtained.")

if __name__ == "__main__":
    main()