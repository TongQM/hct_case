#!/usr/bin/env python3
"""
Simulation Framework for Service Design Evaluation
Compares DRO (robust) vs Non-DRO service designs across different problem sizes.
Tracks provider costs (linehaul, ODD, travel) and user costs (wait time, travel time).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt
from lib.algorithm import Partition
from lib.data import GeoData
import networkx as nx
import time
import argparse
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

class ToyGeoData(GeoData):
    """Toy geographic data for simulation testing with fixed service region"""
    
    def __init__(self, n_blocks, grid_size, service_region_miles=10.0, seed=42):
        np.random.seed(seed)
        self.n_blocks = n_blocks
        self.grid_size = grid_size
        self.service_region_miles = service_region_miles  # Fixed service region size
        self.miles_per_grid_unit = service_region_miles / grid_size  # Resolution scaling
        self.short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
        
        # Create block graph using block IDs as nodes
        self.G = nx.Graph()
        self.block_to_coord = {i: (i // grid_size, i % grid_size) for i in range(n_blocks)}
        
        # Add all block IDs as nodes
        for block_id in self.short_geoid_list:
            self.G.add_node(block_id)
        
        # Add edges between adjacent blocks
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.G.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        # Create block graph for contiguity
        self.block_graph = nx.Graph()
        for i in range(n_blocks):
            self.block_graph.add_node(self.short_geoid_list[i])
        
        for i in range(n_blocks):
            for j in range(i+1, n_blocks):
                coord1 = self.block_to_coord[i]
                coord2 = self.block_to_coord[j]
                if abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) == 1:
                    self.block_graph.add_edge(self.short_geoid_list[i], self.short_geoid_list[j])
        
        # Set attributes
        self.areas = {block_id: 1.0 for block_id in self.short_geoid_list}
        self.gdf = None
    
    def get_dist(self, block1, block2):
        if block1 == block2:
            return 0.0
        try:
            i1 = self.short_geoid_list.index(block1)
            i2 = self.short_geoid_list.index(block2)
            coord1 = self.block_to_coord[i1]
            coord2 = self.block_to_coord[i2]
            # Convert grid distance to miles
            grid_distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
            return grid_distance * self.miles_per_grid_unit
        except:
            return float('inf')
    
    def get_coord_in_miles(self, block_id):
        """Get block center coordinates in miles"""
        try:
            idx = self.short_geoid_list.index(block_id)
            grid_coord = self.block_to_coord[idx]
            # Convert to miles (block centers are at +0.5 offset)
            x_miles = (grid_coord[1] + 0.5) * self.miles_per_grid_unit
            y_miles = (grid_coord[0] + 0.5) * self.miles_per_grid_unit
            return (x_miles, y_miles)
        except:
            return (0.0, 0.0)
    
    def get_area(self, block_id):
        return 1.0
    
    def get_K(self, block_id):
        return 2.0

@dataclass
class ServiceDesign:
    """Service design specification"""
    depot_id: str
    district_roots: List[str]
    assignment: np.ndarray
    dispatch_intervals: Dict[str, float]  # T_star for each district
    design_type: str  # "DRO" or "Non-DRO"
    optimization_objective: float

@dataclass
class SimulationResults:
    """Results from one simulation trial"""
    provider_cost: float
    user_cost: float
    linehaul_cost: float
    odd_cost: float
    travel_cost: float
    max_wait_time: float
    max_travel_time: float
    total_cost: float

def create_true_distribution_sampler(grid_size: int, seed: int = 42):
    """Create sampler for true truncated mixed-Gaussian distribution"""
    np.random.seed(seed)
    
    # Define three cluster centers (scaled to grid size) 
    cluster_centers = [
        (grid_size * 0.25, grid_size * 0.25),  # Top-left cluster
        (grid_size * 0.75, grid_size * 0.25),  # Top-right cluster  
        (grid_size * 0.50, grid_size * 0.75)   # Bottom-center cluster
    ]
    cluster_weights = [0.4, 0.35, 0.25]
    cluster_sigmas = [grid_size * 0.18, grid_size * 0.20, grid_size * 0.15]
    
    def sample_from_true_distribution(n_samples: int, random_seed: Optional[int] = None):
        """Sample demand points from true truncated mixed-Gaussian distribution"""
        if random_seed is not None:
            np.random.seed(random_seed)
            
        samples = []
        for _ in range(n_samples):
            # Choose cluster according to weights
            cluster_idx = np.random.choice(len(cluster_centers), p=cluster_weights)
            center = cluster_centers[cluster_idx]
            sigma = cluster_sigmas[cluster_idx]
            
            # Sample from chosen Gaussian cluster
            x = np.random.normal(center[0], sigma)
            y = np.random.normal(center[1], sigma)
            
            # Truncate to grid boundaries
            x = np.clip(x, 0, grid_size - 1)
            y = np.clip(y, 0, grid_size - 1)
            
            samples.append((x, y))
        
        return samples
    
    return sample_from_true_distribution

def create_nominal_distribution(grid_size: int, n_samples: int = 100, seed: int = 42) -> Dict[str, float]:
    """
    Create nominal distribution by sampling from true distribution and aggregating by blocks
    
    Returns:
    - nominal_prob: Block-based nominal distribution (same for both DRO and Non-DRO)
    """
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
    
    return nominal_prob

def approximate_tsp_length(points: List[Tuple[float, float]]) -> float:
    """
    Approximate TSP tour length using MST heuristic (Christofides bound)
    Returns: approximate tour length through all points
    """
    if len(points) <= 1:
        return 0.0
    
    if len(points) == 2:
        x1, y1 = points[0]
        x2, y2 = points[1]
        return 2.0 * np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    # Compute distance matrix
    coords = np.array(points)
    dist_matrix = squareform(pdist(coords, 'euclidean'))
    
    # Use MST heuristic: TSP ≈ 2 * MST (within factor of 2)
    mst = minimum_spanning_tree(dist_matrix)
    mst_length = mst.sum()
    
    # Christofides bound: TSP ≤ 1.5 * MST for metric spaces
    # We use 2 * MST as a simpler upper bound
    tsp_approx = 2.0 * mst_length
    
    return tsp_approx

def sample_demand_points_in_district(geo_data: ToyGeoData, 
                                   district_blocks: List[str], 
                                   district_demand: int,
                                   true_sampler: callable) -> List[Tuple[float, float]]:
    """
    Sample actual demand points within a district's boundaries
    """
    if district_demand == 0:
        return []
    
    # Sample many points and filter to district
    oversample_factor = 5  # Sample more points than needed
    candidate_samples = true_sampler(district_demand * oversample_factor)
    
    district_points = []
    grid_size = geo_data.grid_size
    miles_per_unit = geo_data.miles_per_grid_unit
    
    # Get district block boundaries
    district_block_coords = set()
    for block_id in district_blocks:
        idx = geo_data.short_geoid_list.index(block_id)
        grid_coord = geo_data.block_to_coord[idx]
        district_block_coords.add(grid_coord)
    
    for x_grid, y_grid in candidate_samples:
        # Check if point falls within district blocks
        block_row = int(np.clip(np.floor(x_grid), 0, grid_size - 1))
        block_col = int(np.clip(np.floor(y_grid), 0, grid_size - 1))
        
        if (block_row, block_col) in district_block_coords:
            # Convert to miles
            x_miles = x_grid * miles_per_unit
            y_miles = y_grid * miles_per_unit
            district_points.append((x_miles, y_miles))
            
            if len(district_points) >= district_demand:
                break
    
    # If we don't have enough points, pad with block centers
    while len(district_points) < district_demand:
        # Use district block centers as fallback
        for block_id in district_blocks:
            if len(district_points) >= district_demand:
                break
            x_miles, y_miles = geo_data.get_coord_in_miles(block_id)
            district_points.append((x_miles, y_miles))
    
    return district_points[:district_demand]

def create_odd_features(grid_size: int, seed: int = 42) -> Tuple[Dict[str, np.ndarray], callable]:
    """Create ODD features and J function"""
    np.random.seed(seed)
    n_blocks = grid_size * grid_size
    short_geoid_list = [f"BLK{i:03d}" for i in range(n_blocks)]
    
    Omega_dict = {}
    
    # ODD feature parameters
    alpha1, beta1 = 2.5, 1.8
    scale1, shift1 = 3.0, 1.5
    
    # Feature 2: Mixture of two modes with spatial correlation
    mode1_center = (grid_size * 0.3, grid_size * 0.3)
    mode2_center = (grid_size * 0.7, grid_size * 0.7)
    mode1_weight, mode2_weight = 0.6, 0.4
    
    for i, block_id in enumerate(short_geoid_list):
        row = i // grid_size
        col = i % grid_size
        
        # Feature 1: Spatially correlated Beta distribution
        normalized_dist = np.sqrt((row/grid_size)**2 + (col/grid_size)**2) / np.sqrt(2)
        beta_sample = np.random.beta(alpha1, beta1)
        omega1 = shift1 + scale1 * (beta_sample + 0.3 * normalized_dist)
        
        # Feature 2: Bimodal spatial distribution
        dist_to_mode1 = np.sqrt((row - mode1_center[0])**2 + (col - mode1_center[1])**2)
        dist_to_mode2 = np.sqrt((row - mode2_center[0])**2 + (col - mode2_center[1])**2)
        
        mode1_prob = mode1_weight * np.exp(-dist_to_mode1 / (grid_size * 0.25))
        mode2_prob = mode2_weight * np.exp(-dist_to_mode2 / (grid_size * 0.25))
        total_mode_prob = mode1_prob + mode2_prob
        
        if total_mode_prob > 0:
            mode1_normalized = mode1_prob / total_mode_prob
        else:
            mode1_normalized = 0.5
            
        # Sample from appropriate mode
        if np.random.uniform() < mode1_normalized:
            omega2 = 2.2 + 0.8 * np.random.gamma(2.0, 0.5)  # Mode 1: lower values
        else:
            omega2 = 3.8 + 0.6 * np.random.gamma(1.5, 0.7)  # Mode 2: higher values
            
        omega2 += 0.15 * np.random.normal(0, 1)
        omega2 = max(0.5, omega2)
        
        Omega_dict[block_id] = np.array([omega1, omega2])
    
    def J_function(omega_vector):
        if hasattr(omega_vector, '__len__') and len(omega_vector) >= 2:
            return 0.6 * omega_vector[0] + 0.4 * omega_vector[1]
        return 0.0
    
    return Omega_dict, J_function

def optimize_service_design(geo_data: ToyGeoData, 
                           nominal_prob: Dict[str, float],  # Block-based nominal distribution
                           Omega_dict: Dict[str, np.ndarray],
                           J_function: callable,
                           num_districts: int,
                           design_type: str,
                           wasserstein_radius: float = 0.1) -> ServiceDesign:
    """Optimize service design using either DRO or Non-DRO approach"""
    
    if design_type == "DRO":
        # Use robust optimization with Wasserstein radius
        epsilon = wasserstein_radius
        print(f"   🛡️  Optimizing {design_type} design (ε={epsilon})")
    else:
        # Use nominal distribution directly
        epsilon = 0.0  # No robustness
        print(f"   📊 Optimizing {design_type} design (nominal)")
    
    # Create partition instance
    # Both DRO and Non-DRO use same nominal, difference is epsilon (robustness parameter)
    partition = Partition(geo_data, num_districts=num_districts, 
                         prob_dict=nominal_prob, epsilon=epsilon)
    
    # Run enhanced heuristics
    depot, centers, assignment, obj_val, district_info = partition.random_search(
        max_iters=50,  # Reduced for faster simulation
        prob_dict=nominal_prob,
        Lambda=25.0,
        wr=1.0,
        wv=10.0,
        beta=0.7120,
        Omega_dict=Omega_dict,
        J_function=J_function
    )
    
    # Extract dispatch intervals from district_info
    dispatch_intervals = {}
    for cost, root, K_i, F_i, T_star in district_info:
        dispatch_intervals[root] = T_star
    
    return ServiceDesign(
        depot_id=depot,
        district_roots=centers,
        assignment=assignment,
        dispatch_intervals=dispatch_intervals,
        design_type=design_type,
        optimization_objective=obj_val
    )

def simulate_one_day_operation(geo_data: ToyGeoData,
                              service_design: ServiceDesign,
                              true_sampler: callable,
                              Omega_dict: Dict[str, np.ndarray],
                              J_function: callable,
                              daily_demand_rate: float = 100.0) -> SimulationResults:
    """Simulate one day's operation of the service design using true distribution"""
    
    block_ids = geo_data.short_geoid_list
    N = len(block_ids)
    grid_size = geo_data.grid_size
    
    # Generate daily demand by sampling from true distribution
    demand_samples = true_sampler(int(daily_demand_rate))
    
    # Aggregate demand by block
    daily_demands = {block_id: 0 for block_id in block_ids}
    total_daily_demand = 0
    
    for x, y in demand_samples:
        # Determine which block this demand falls into
        block_row = int(np.clip(np.floor(x), 0, grid_size - 1))
        block_col = int(np.clip(np.floor(y), 0, grid_size - 1))
        block_idx = block_row * grid_size + block_col
        block_id = block_ids[block_idx]
        
        daily_demands[block_id] += 1
        total_daily_demand += 1
    
    # Compute costs for each district
    provider_cost = 0.0
    user_cost = 0.0
    linehaul_cost = 0.0
    odd_cost = 0.0
    travel_cost = 0.0
    max_wait_time = 0.0
    max_travel_time = 0.0
    
    for root_id in service_design.district_roots:
        root_idx = block_ids.index(root_id)
        
        # Get blocks assigned to this district
        assigned_blocks = []
        district_demand = 0
        for i in range(N):
            if round(service_design.assignment[i, root_idx]) == 1:
                block_id = block_ids[i]
                assigned_blocks.append(block_id)
                district_demand += daily_demands[block_id]
        
        if district_demand == 0:
            continue
            
        # Dispatch interval for this district
        T_star = service_design.dispatch_intervals[root_id]
        
        # Number of dispatches needed
        num_dispatches = max(1, int(np.ceil(24.0 / T_star)))  # Dispatches per day
        demand_per_dispatch = district_demand / num_dispatches
        
        # Provider costs
        # 1. Linehaul cost: fixed cost from depot to closest block in district
        min_depot_dist = min(geo_data.get_dist(service_design.depot_id, block_id) 
                            for block_id in assigned_blocks)
        dispatch_linehaul = min_depot_dist * num_dispatches
        linehaul_cost += dispatch_linehaul
        
        # 2. ODD cost: based on district ODD features
        district_omega = np.zeros(2)
        for block_id in assigned_blocks:
            if block_id in Omega_dict:
                # Element-wise maximum for district ODD features
                district_omega = np.maximum(district_omega, Omega_dict[block_id])
        
        district_odd_cost = J_function(district_omega) * num_dispatches
        odd_cost += district_odd_cost
        
        # 3. Travel cost: realistic TSP-based vehicle routing within district
        district_travel = 0.0
        for dispatch in range(num_dispatches):
            # Sample actual demand points for this dispatch
            dispatch_demand = int(np.ceil(demand_per_dispatch))
            if dispatch_demand > 0:
                # Get dispatch-specific sampler with unique seed
                dispatch_seed = hash((grid_size, root_id, dispatch, daily_demand_rate)) % 1000000
                def dispatch_sampler(n):
                    return true_sampler(n, dispatch_seed)
                
                demand_points = sample_demand_points_in_district(
                    geo_data, assigned_blocks, dispatch_demand, dispatch_sampler
                )
                
                if demand_points:
                    # Approximate TSP tour length through demand points
                    tsp_length = approximate_tsp_length(demand_points)
                    district_travel += tsp_length
        
        travel_cost += district_travel
        
        # User costs
        # 1. Wait time: maximum dispatch interval (worst case)
        district_wait = T_star
        max_wait_time = max(max_wait_time, district_wait)
        
        # 2. Travel time: linehaul + in-district travel
        # Linehaul: half distance from depot to district center
        linehaul_travel_time = min_depot_dist / 2.0
        
        # In-district: maximum distance from root to any block in district
        max_in_district_dist = max(
            geo_data.get_dist(root_id, block_id) 
            for block_id in assigned_blocks
        )
        
        district_travel_time = linehaul_travel_time + max_in_district_dist
        max_travel_time = max(max_travel_time, district_travel_time)
    
    provider_cost = linehaul_cost + odd_cost + travel_cost
    user_cost = max_wait_time + max_travel_time
    total_cost = provider_cost + user_cost
    
    return SimulationResults(
        provider_cost=provider_cost,
        user_cost=user_cost,
        linehaul_cost=linehaul_cost,
        odd_cost=odd_cost,
        travel_cost=travel_cost,
        max_wait_time=max_wait_time,
        max_travel_time=max_travel_time,
        total_cost=total_cost
    )

def run_simulation_study(grid_sizes: List[int], 
                        num_trials: int = 5,
                        num_districts: int = 3,
                        wasserstein_radius: float = 0.1,
                        n_samples: int = 100) -> pd.DataFrame:
    """Run comprehensive simulation study across different grid sizes"""
    
    print("=" * 80)
    print("SERVICE DESIGN SIMULATION STUDY")
    print("=" * 80)
    print(f"📊 Grid sizes: {grid_sizes}")
    print(f"🔁 Trials per configuration: {num_trials}")
    print(f"🏛️  Districts: {num_districts}")
    print(f"🛡️  Wasserstein radius: {wasserstein_radius}")
    print(f"📍 Nominal samples: {n_samples}")
    print()
    print("🎯 DISTRIBUTIONAL FRAMEWORK:")
    print("   • True: Truncated mixed-Gaussian (continuous)")
    print("   • Observed: Samples from true distribution")  
    print("   • Nominal: Block aggregation of samples (same for both)")
    print("   • DRO: ε > 0 (robust against worst-case in Wasserstein ball)")
    print("   • Non-DRO: ε = 0 (optimizes against nominal directly)")
    print("   • Simulation: New samples from true distribution")
    print()
    
    results = []
    
    for grid_size in grid_sizes:
        n_blocks = grid_size * grid_size
        print(f"\\n🎯 Testing {grid_size}×{grid_size} grid ({n_blocks} blocks)")
        print("-" * 60)
        
        # Create problem instance with fixed 10x10 mile service region
        geo_data = ToyGeoData(n_blocks, grid_size, service_region_miles=10.0)
        
        # Create true distribution sampler
        true_sampler = create_true_distribution_sampler(grid_size)
        
        # Create nominal distribution from samples (same for both DRO and Non-DRO)
        nominal_prob = create_nominal_distribution(
            grid_size, n_samples, seed=42 + grid_size  # Different seed per grid size
        )
        
        # Create ODD features
        Omega_dict, J_function = create_odd_features(grid_size)
        
        # Optimize both service designs using same nominal but different epsilon
        dro_design = optimize_service_design(
            geo_data, nominal_prob, Omega_dict, J_function, 
            num_districts, "DRO", wasserstein_radius
        )
        
        non_dro_design = optimize_service_design(
            geo_data, nominal_prob, Omega_dict, J_function, 
            num_districts, "Non-DRO", 0.0
        )
        
        print(f"   ✅ DRO design optimized (obj: {dro_design.optimization_objective:.3f})")
        print(f"   ✅ Non-DRO design optimized (obj: {non_dro_design.optimization_objective:.3f})")
        
        # Run simulation trials
        for trial in range(num_trials):
            print(f"   🎲 Trial {trial + 1}/{num_trials}")
            
            # Simulate both designs using true distribution
            # Each trial gets different random seed for demand generation
            trial_seed = 1000 + trial * 100 + grid_size
            
            def trial_sampler(n_samples, random_seed=None):
                # Use trial_seed if no specific seed provided
                seed_to_use = random_seed if random_seed is not None else trial_seed
                return true_sampler(n_samples, seed_to_use)
            
            dro_result = simulate_one_day_operation(
                geo_data, dro_design, trial_sampler, Omega_dict, J_function
            )
            
            non_dro_result = simulate_one_day_operation(
                geo_data, non_dro_design, trial_sampler, Omega_dict, J_function
            )
            
            # Record results
            for design_type, sim_result in [("DRO", dro_result), ("Non-DRO", non_dro_result)]:
                results.append({
                    'blocks': n_blocks,
                    'grid_size': grid_size,
                    'trial': trial + 1,
                    'design_type': design_type,
                    'total_cost': sim_result.total_cost,
                    'provider_cost': sim_result.provider_cost,
                    'user_cost': sim_result.user_cost,
                    'linehaul_cost': sim_result.linehaul_cost,
                    'odd_cost': sim_result.odd_cost,
                    'travel_cost': sim_result.travel_cost,
                    'max_wait_time': sim_result.max_wait_time,
                    'max_travel_time': sim_result.max_travel_time
                })
    
    return pd.DataFrame(results)

def analyze_and_visualize_results(df: pd.DataFrame, service_designs_info: Dict = None) -> None:
    """Analyze simulation results and create summary table and visualizations"""
    
    print("\\n" + "=" * 80)
    print("SIMULATION RESULTS ANALYSIS")
    print("=" * 80)
    
    # Create summary statistics
    summary = df.groupby(['blocks', 'design_type']).agg({
        'total_cost': ['mean', 'std'],
        'provider_cost': ['mean', 'std'], 
        'user_cost': ['mean', 'std'],
        'linehaul_cost': ['mean', 'std'],
        'odd_cost': ['mean', 'std'],
        'travel_cost': ['mean', 'std']
    }).round(1)
    
    # Flatten column names
    summary.columns = ['_'.join(col).strip() for col in summary.columns]
    summary = summary.reset_index()
    
    # Create comparison table similar to the research table
    print("\\n📊 COSTS OF SERVICE DESIGNS (Mean ± Std)")
    print("=" * 80)
    
    comparison_table = []
    for blocks in sorted(df['blocks'].unique()):
        dro_data = summary[(summary['blocks'] == blocks) & (summary['design_type'] == 'DRO')]
        non_dro_data = summary[(summary['blocks'] == blocks) & (summary['design_type'] == 'Non-DRO')]
        
        if not dro_data.empty and not non_dro_data.empty:
            comparison_table.append({
                'Blocks': blocks,
                'DRO Cost': f"{dro_data['total_cost_mean'].iloc[0]:.1f} ± {dro_data['total_cost_std'].iloc[0]:.1f}",
                'Non-DRO Cost': f"{non_dro_data['total_cost_mean'].iloc[0]:.1f} ± {non_dro_data['total_cost_std'].iloc[0]:.1f}",
                'PROV (DRO)': f"{dro_data['provider_cost_mean'].iloc[0]:.1f} ± {dro_data['provider_cost_std'].iloc[0]:.1f}",
                'USER (DRO)': f"{dro_data['user_cost_mean'].iloc[0]:.1f} ± {dro_data['user_cost_std'].iloc[0]:.1f}",
                'Linehaul': f"{dro_data['linehaul_cost_mean'].iloc[0]:.1f} ± {dro_data['linehaul_cost_std'].iloc[0]:.1f}",
                'ODD': f"{dro_data['odd_cost_mean'].iloc[0]:.1f} ± {dro_data['odd_cost_std'].iloc[0]:.1f}",
                'Travel': f"{dro_data['travel_cost_mean'].iloc[0]:.1f} ± {dro_data['travel_cost_std'].iloc[0]:.1f}"
            })
    
    comparison_df = pd.DataFrame(comparison_table)
    print(comparison_df.to_string(index=False))
    
    # Create comprehensive visualizations
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('Service Design Simulation Results with TSP Travel Costs\\nFixed 10×10 Mile Service Region', 
                fontsize=16, fontweight='bold')
    
    # 1. Total Cost Comparison
    ax1 = axes[0, 0]
    blocks_list = sorted(df['blocks'].unique())
    dro_means = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'DRO')]['total_cost_mean'].iloc[0] for b in blocks_list]
    non_dro_means = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'Non-DRO')]['total_cost_mean'].iloc[0] for b in blocks_list]
    dro_stds = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'DRO')]['total_cost_std'].iloc[0] for b in blocks_list]
    non_dro_stds = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'Non-DRO')]['total_cost_std'].iloc[0] for b in blocks_list]
    
    x = np.arange(len(blocks_list))
    width = 0.35
    
    ax1.bar(x - width/2, dro_means, width, yerr=dro_stds, label='DRO', alpha=0.8, capsize=5)
    ax1.bar(x + width/2, non_dro_means, width, yerr=non_dro_stds, label='Non-DRO', alpha=0.8, capsize=5)
    ax1.set_xlabel('Number of Blocks')
    ax1.set_ylabel('Total Cost')
    ax1.set_title('Total Cost Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(blocks_list)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Provider vs User Cost Breakdown
    ax2 = axes[0, 1]
    dro_prov = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'DRO')]['provider_cost_mean'].iloc[0] for b in blocks_list]
    dro_user = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'DRO')]['user_cost_mean'].iloc[0] for b in blocks_list]
    
    ax2.bar(x, dro_prov, width, label='Provider Cost (DRO)', alpha=0.8)
    ax2.bar(x, dro_user, width, bottom=dro_prov, label='User Cost (DRO)', alpha=0.8)
    ax2.set_xlabel('Number of Blocks')
    ax2.set_ylabel('Cost')
    ax2.set_title('Cost Breakdown (DRO Design)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(blocks_list)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Cost Components (Enhanced with Travel Cost)
    ax3 = axes[1, 0]
    linehaul_means = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'DRO')]['linehaul_cost_mean'].iloc[0] for b in blocks_list]
    odd_means = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'DRO')]['odd_cost_mean'].iloc[0] for b in blocks_list]
    travel_means = [summary[(summary['blocks'] == b) & (summary['design_type'] == 'DRO')]['travel_cost_mean'].iloc[0] for b in blocks_list]
    
    ax3.plot(blocks_list, linehaul_means, 'o-', label='Linehaul Cost', linewidth=2, markersize=8)
    ax3.plot(blocks_list, odd_means, 's-', label='ODD Cost', linewidth=2, markersize=8)
    ax3.plot(blocks_list, travel_means, '^-', label='TSP Travel Cost', linewidth=2, markersize=8)
    ax3.set_xlabel('Number of Blocks')
    ax3.set_ylabel('Cost (Miles)')
    ax3.set_title('Cost Component Scaling (DRO)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Robustness Benefit
    ax4 = axes[1, 1]
    cost_diff = [non_dro_means[i] - dro_means[i] for i in range(len(blocks_list))]
    percent_improvement = [(non_dro_means[i] - dro_means[i]) / non_dro_means[i] * 100 for i in range(len(blocks_list))]
    
    ax4.bar(x, percent_improvement, alpha=0.8, color='green')
    ax4.set_xlabel('Number of Blocks')
    ax4.set_ylabel('Cost Reduction (%)')
    ax4.set_title('DRO Benefit vs Non-DRO')
    ax4.set_xticks(x)
    ax4.set_xticklabels(blocks_list)
    ax4.grid(True, alpha=0.3)
    
    # Add percentage labels on bars
    for i, v in enumerate(percent_improvement):
        ax4.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 5. Travel vs Linehaul Cost Comparison
    ax5 = axes[2, 0]
    grid_sizes = [int(np.sqrt(b)) for b in blocks_list]  # Convert blocks to grid sizes
    linehaul_per_mile = [linehaul_means[i] / (10.0) for i in range(len(blocks_list))]  # Normalize by service region
    travel_per_dispatch = [travel_means[i] / max(1, b/10) for i, b in enumerate(blocks_list)]  # Rough normalization
    
    ax5.scatter(grid_sizes, linehaul_per_mile, s=100, alpha=0.7, label='Linehaul/Mile', marker='o')
    ax5.scatter(grid_sizes, travel_per_dispatch, s=100, alpha=0.7, label='Travel/Dispatch', marker='^')
    ax5.set_xlabel('Grid Resolution')
    ax5.set_ylabel('Normalized Cost')
    ax5.set_title('Travel vs Linehaul Scaling')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Service Region Resolution Impact
    ax6 = axes[2, 1]
    block_density = [b / 100.0 for b in blocks_list]  # Blocks per square mile (10x10 = 100 sq miles)
    total_cost_per_block = [dro_means[i] / blocks_list[i] for i in range(len(blocks_list))]
    
    ax6.plot(block_density, total_cost_per_block, 'o-', linewidth=2, markersize=8, color='purple')
    ax6.set_xlabel('Block Density (blocks/sq mile)')
    ax6.set_ylabel('Cost per Block')
    ax6.set_title('Cost Efficiency vs Resolution')
    ax6.grid(True, alpha=0.3)
    
    # Add trend annotation
    if len(block_density) > 1:
        slope = (total_cost_per_block[-1] - total_cost_per_block[0]) / (block_density[-1] - block_density[0])
        ax6.text(0.05, 0.95, f'Trend: {slope:.2f} cost/density', transform=ax6.transAxes, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('service_design_simulation_results.png', dpi=300, bbox_inches='tight')
    print(f"\\n📁 Saved comprehensive visualization as 'service_design_simulation_results.png'")
    
    # Save detailed results to CSV
    df.to_csv('simulation_results_detailed.csv', index=False)
    comparison_df.to_csv('simulation_results_summary.csv', index=False)
    print(f"📁 Saved detailed results to 'simulation_results_detailed.csv'")
    print(f"📁 Saved summary table to 'simulation_results_summary.csv'")

def main():
    """Main function for running simulation study"""
    parser = argparse.ArgumentParser(description='Service Design Simulation Study')
    parser.add_argument('--grid_sizes', nargs='+', type=int, default=[3, 4, 5, 6, 7, 8], 
                       help='Grid sizes to test (default: 3 4 5 6 7 8)')
    parser.add_argument('--trials', type=int, default=5, help='Number of trials per configuration (default: 5)')
    parser.add_argument('--districts', type=int, default=3, help='Number of districts (default: 3)')
    parser.add_argument('--wasserstein_radius', type=float, default=0.1, help='Wasserstein radius for DRO (default: 0.1)')
    parser.add_argument('--n_samples', type=int, default=100, help='Number of samples for nominal distribution (default: 100)')
    
    args = parser.parse_args()
    
    # Run simulation study
    results_df = run_simulation_study(
        grid_sizes=args.grid_sizes,
        num_trials=args.trials,
        num_districts=args.districts,
        wasserstein_radius=args.wasserstein_radius,
        n_samples=args.n_samples
    )
    
    # Analyze and visualize results
    analyze_and_visualize_results(results_df)
    
    print(f"\\n🎉 Simulation study completed successfully!")
    print(f"   📊 Tested {len(args.grid_sizes)} grid sizes with {args.trials} trials each")
    print(f"   🛡️  Compared DRO vs Non-DRO service designs")
    print(f"   📈 Generated comprehensive cost analysis and visualizations")

if __name__ == "__main__":
    main()