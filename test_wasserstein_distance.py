#!/usr/bin/env python3
"""
Test script for evaluating Wasserstein distance between true and empirical distributions.

Given a fixed underlying true distribution f, sample n points from f.
For each point, it has a probability to appear or not.
Finally evaluate the empirical distribution from the samples to the true distribution.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
import logging
from typing import Dict, Tuple, List
import random

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_wasserstein_distance_1d(dist1: np.ndarray, dist2: np.ndarray, locations: np.ndarray) -> float:
    """
    Compute 1-dimensional Wasserstein distance between two distributions.
    
    Parameters:
    - dist1: First probability distribution
    - dist2: Second probability distribution  
    - locations: Locations/positions of the support points
    
    Returns:
    - Wasserstein distance
    """
    assert len(dist1) == len(dist2) == len(locations), "All arrays must have same length"
    assert np.isclose(np.sum(dist1), 1.0) and np.isclose(np.sum(dist2), 1.0), "Distributions must sum to 1"
    
    # Sort by locations
    sorted_indices = np.argsort(locations)
    sorted_dist1 = dist1[sorted_indices]
    sorted_dist2 = dist2[sorted_indices]
    sorted_locations = locations[sorted_indices]
    
    # Compute cumulative distributions
    cum1 = np.cumsum(sorted_dist1)
    cum2 = np.cumsum(sorted_dist2)
    
    # Compute Wasserstein distance using the dual formulation
    wasserstein_dist = 0.0
    for i in range(len(sorted_locations) - 1):
        segment_length = sorted_locations[i + 1] - sorted_locations[i]
        wasserstein_dist += abs(cum1[i] - cum2[i]) * segment_length
    
    return wasserstein_dist


def compute_wasserstein_distance_2d(dist1: np.ndarray, dist2: np.ndarray, locations: np.ndarray) -> float:
    """
    Compute 2-dimensional Wasserstein distance using linear programming.
    
    Parameters:
    - dist1: First probability distribution (shape: n)
    - dist2: Second probability distribution (shape: n)
    - locations: 2D locations/positions of the support points (shape: n x 2)
    
    Returns:
    - Wasserstein distance
    """
    n = len(dist1)
    assert len(dist2) == n and locations.shape[0] == n, "All inputs must have same number of points"
    assert np.isclose(np.sum(dist1), 1.0) and np.isclose(np.sum(dist2), 1.0), "Distributions must sum to 1"
    
    # Compute pairwise distances (cost matrix)
    cost_matrix = cdist(locations, locations, metric='euclidean')
    
    # Flatten the cost matrix for linear programming
    c = cost_matrix.flatten()
    
    # Equality constraints: marginal constraints
    # First set: sum over j of x_{i,j} = dist1[i] for all i
    A_eq1 = np.zeros((n, n*n))
    for i in range(n):
        for j in range(n):
            A_eq1[i, i*n + j] = 1
    b_eq1 = dist1
    
    # Second set: sum over i of x_{i,j} = dist2[j] for all j  
    A_eq2 = np.zeros((n, n*n))
    for j in range(n):
        for i in range(n):
            A_eq2[j, i*n + j] = 1
    b_eq2 = dist2
    
    # Combine constraints
    A_eq = np.vstack([A_eq1, A_eq2])
    b_eq = np.hstack([b_eq1, b_eq2])
    
    # Bounds: all variables >= 0
    bounds = [(0, None) for _ in range(n*n)]
    
    # Solve the linear program
    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    
    if not result.success:
        logger.warning("Linear programming solver failed, returning large value")
        return float('inf')
    
    return result.fun


class WassersteinTester:
    """Test class for Wasserstein distance evaluation with sampling probabilities."""
    
    def __init__(self, n_points: int = 100, dimension: int = 2, seed: int = 42):
        """
        Initialize the tester.
        
        Parameters:
        - n_points: Number of support points
        - dimension: Dimension of the space (1 or 2)
        - seed: Random seed for reproducibility
        """
        self.n_points = n_points
        self.dimension = dimension
        np.random.seed(seed)
        random.seed(seed)
        
        # Generate random support points
        if dimension == 1:
            self.locations = np.sort(np.random.uniform(0, 10, n_points))
        elif dimension == 2:
            self.locations = np.random.uniform(0, 10, (n_points, 2))
        else:
            raise ValueError("Only 1D and 2D supported")
            
        # Generate true distribution (normalized random weights)
        raw_weights = np.random.exponential(1.0, n_points)
        self.true_distribution = raw_weights / np.sum(raw_weights)
        
        logger.info(f"Initialized {dimension}D tester with {n_points} points")
    
    def generate_empirical_distribution(self, n_samples: int, appearance_prob: float = 0.8) -> np.ndarray:
        """
        Generate empirical distribution by sampling from true distribution with appearance probabilities.
        
        Mathematical formulation:
        1. Sample n_samples points according to true_distribution p
        2. Each sampled point appears independently with probability appearance_prob
        3. Empirical distribution: p̂_i = C_i / (total_appeared_samples)
        
        This is the standard unbiased empirical distribution estimator.
        
        Parameters:
        - n_samples: Number of samples to draw
        - appearance_prob: Probability that each sampled point actually appears
        
        Returns:
        - Empirical distribution (properly normalized probability distribution)
        """
        # Sample points according to true distribution
        sampled_indices = np.random.choice(
            self.n_points, 
            size=n_samples, 
            p=self.true_distribution
        )
        
        # Apply appearance probability: each sampled point appears with given probability
        appeared_indices = []
        for idx in sampled_indices:
            if np.random.random() < appearance_prob:
                appeared_indices.append(idx)
        
        # Count appearances at each location
        empirical_counts = np.zeros(self.n_points)
        for idx in appeared_indices:
            empirical_counts[idx] += 1
        
        # Standard empirical distribution: normalize by total appeared samples
        total_appeared = len(appeared_indices)
        if total_appeared > 0:
            empirical_distribution = empirical_counts / total_appeared
        else:
            # If no points appeared, return uniform distribution
            empirical_distribution = np.ones(self.n_points) / self.n_points
        
        logger.debug(f"Generated empirical distribution from {total_appeared} appeared points out of {n_samples} samples")
        return empirical_distribution
    
    def compute_wasserstein_distance(self, empirical_dist: np.ndarray) -> float:
        """
        Compute Wasserstein distance between true and empirical distributions.
        
        Parameters:
        - empirical_dist: Empirical distribution
        
        Returns:
        - Wasserstein distance
        """
        if self.dimension == 1:
            return compute_wasserstein_distance_1d(
                self.true_distribution, 
                empirical_dist, 
                self.locations
            )
        elif self.dimension == 2:
            return compute_wasserstein_distance_2d(
                self.true_distribution, 
                empirical_dist, 
                self.locations
            )
        else:
            raise ValueError("Only 1D and 2D supported")
    
    def run_experiment(self, n_samples_list: List[int], appearance_probs: List[float], 
                      n_repetitions: int = 10) -> Dict:
        """
        Run comprehensive experiment varying sample sizes and appearance probabilities.
        
        Parameters:
        - n_samples_list: List of sample sizes to test
        - appearance_probs: List of appearance probabilities to test
        - n_repetitions: Number of repetitions for each configuration
        
        Returns:
        - Dictionary with experimental results
        """
        results = {
            'n_samples': [],
            'appearance_prob': [],
            'wasserstein_distance': [],
            'repetition': []
        }
        
        total_experiments = len(n_samples_list) * len(appearance_probs) * n_repetitions
        experiment_count = 0
        
        for n_samples in n_samples_list:
            for app_prob in appearance_probs:
                for rep in range(n_repetitions):
                    experiment_count += 1
                    logger.info(f"Experiment {experiment_count}/{total_experiments}: "
                              f"n_samples={n_samples}, app_prob={app_prob:.2f}, rep={rep+1}")
                    
                    # Generate empirical distribution
                    empirical_dist = self.generate_empirical_distribution(n_samples, app_prob)
                    
                    # Compute Wasserstein distance
                    wasserstein_dist = self.compute_wasserstein_distance(empirical_dist)
                    
                    # Store results
                    results['n_samples'].append(n_samples)
                    results['appearance_prob'].append(app_prob)
                    results['wasserstein_distance'].append(wasserstein_dist)
                    results['repetition'].append(rep)
        
        return results
    
    def plot_results(self, results: Dict, save_path: str = None):
        """
        Plot experimental results.
        
        Parameters:
        - results: Results dictionary from run_experiment
        - save_path: Optional path to save the plot
        """
        import pandas as pd
        
        df = pd.DataFrame(results)
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Wasserstein distance vs sample size (averaged over appearance probs)
        df_grouped = df.groupby('n_samples')['wasserstein_distance'].agg(['mean', 'std']).reset_index()
        ax1.errorbar(df_grouped['n_samples'], df_grouped['mean'], yerr=df_grouped['std'], 
                    marker='o', capsize=5)
        ax1.set_xlabel('Number of Samples')
        ax1.set_ylabel('Wasserstein Distance')
        ax1.set_title('Wasserstein Distance vs Sample Size')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Wasserstein distance vs appearance probability (averaged over sample sizes)
        df_grouped = df.groupby('appearance_prob')['wasserstein_distance'].agg(['mean', 'std']).reset_index()
        ax2.errorbar(df_grouped['appearance_prob'], df_grouped['mean'], yerr=df_grouped['std'], 
                    marker='o', capsize=5)
        ax2.set_xlabel('Appearance Probability')
        ax2.set_ylabel('Wasserstein Distance')
        ax2.set_title('Wasserstein Distance vs Appearance Probability')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Heatmap of Wasserstein distance
        pivot_df = df.groupby(['n_samples', 'appearance_prob'])['wasserstein_distance'].mean().unstack()
        im = ax3.imshow(pivot_df.values, aspect='auto', cmap='viridis')
        ax3.set_xticks(range(len(pivot_df.columns)))
        ax3.set_xticklabels([f'{x:.2f}' for x in pivot_df.columns])
        ax3.set_yticks(range(len(pivot_df.index)))
        ax3.set_yticklabels(pivot_df.index)
        ax3.set_xlabel('Appearance Probability')
        ax3.set_ylabel('Number of Samples')
        ax3.set_title('Wasserstein Distance Heatmap')
        plt.colorbar(im, ax=ax3)
        
        # Plot 4: Distribution comparison example
        if self.dimension == 1:
            empirical_example = self.generate_empirical_distribution(1000, 0.7)
            ax4.plot(self.locations, self.true_distribution, 'b-', label='True Distribution', linewidth=2)
            ax4.plot(self.locations, empirical_example, 'r--', label='Empirical Distribution', linewidth=2)
            ax4.set_xlabel('Location')
            ax4.set_ylabel('Probability Density')
            ax4.set_title('Example: True vs Empirical Distribution (1D)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            # For 2D, show scatter plot of support points
            scatter = ax4.scatter(self.locations[:, 0], self.locations[:, 1], 
                                c=self.true_distribution, cmap='viridis', s=50)
            ax4.set_xlabel('X Location')
            ax4.set_ylabel('Y Location')
            ax4.set_title('Support Points (2D, colored by true distribution)')
            plt.colorbar(scatter, ax=ax4)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """Main function to run the Wasserstein distance test."""
    logger.info("Starting Wasserstein distance evaluation test")
    
    # Test parameters
    n_points = 50
    dimension = 2  # Change to 1 for 1D test
    n_samples_list = [50, 100, 200, 500, 1000]
    appearance_probs = [0.3, 0.5, 0.7, 0.9, 1.0]  # Include normal sampling (no appearance filter)
    n_repetitions = 5
    
    # Create tester
    tester = WassersteinTester(n_points=n_points, dimension=dimension, seed=42)
    
    # Run experiment
    logger.info("Running experiment...")
    results = tester.run_experiment(n_samples_list, appearance_probs, n_repetitions)
    
    # Plot results
    logger.info("Plotting results...")
    tester.plot_results(results, save_path=f'./figures/wasserstein_test_{dimension}d.png')
    
    # Print summary statistics
    import pandas as pd
    df = pd.DataFrame(results)
    
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Dimension: {dimension}D")
    print(f"Number of support points: {n_points}")
    print(f"Sample sizes tested: {n_samples_list}")
    print(f"Appearance probabilities tested: {appearance_probs}")
    print(f"Repetitions per configuration: {n_repetitions}")
    print(f"Total experiments: {len(df)}")
    
    print(f"\nWasserstein distance statistics:")
    print(f"  Mean: {df['wasserstein_distance'].mean():.4f}")
    print(f"  Std:  {df['wasserstein_distance'].std():.4f}")
    print(f"  Min:  {df['wasserstein_distance'].min():.4f}")
    print(f"  Max:  {df['wasserstein_distance'].max():.4f}")
    
    # Best and worst configurations
    best_idx = df['wasserstein_distance'].idxmin()
    worst_idx = df['wasserstein_distance'].idxmax()
    
    print(f"\nBest configuration (lowest Wasserstein distance):")
    print(f"  Samples: {df.loc[best_idx, 'n_samples']}, "
          f"App. prob: {df.loc[best_idx, 'appearance_prob']:.2f}, "
          f"Distance: {df.loc[best_idx, 'wasserstein_distance']:.4f}")
    
    print(f"\nWorst configuration (highest Wasserstein distance):")
    print(f"  Samples: {df.loc[worst_idx, 'n_samples']}, "
          f"App. prob: {df.loc[worst_idx, 'appearance_prob']:.2f}, "
          f"Distance: {df.loc[worst_idx, 'wasserstein_distance']:.4f}")
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()