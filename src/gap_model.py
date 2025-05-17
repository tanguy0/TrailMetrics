import numpy as np
from typing import Dict, Tuple
import matplotlib.pyplot as plt


def calculate_normalized_efficiency(
    speeds: np.ndarray,
    elevation_gains: np.ndarray,
    heartrates: np.ndarray,
) -> np.ndarray:
    """
    Calculate normalized efficiency for each split.
    
    Args:
        speeds: Array of speeds (km/h)
        elevation_gains: Array of elevation gains (m/km)
        heartrates: Array of heart rates (bpm)
        
    Returns:
        Array of normalized efficiencies
    """
    # Calculate efficiency (heart rate / speed)
    efficiencies = heartrates / speeds
    
    # Find nearly flat sections (gradient between -1% and 1%)
    flat_mask = (elevation_gains > -10) & (elevation_gains <10)
    
    if np.sum(flat_mask) > 100:
        # Calculate median efficiency for flat sections
        median_flat_efficiency = np.median(efficiencies[flat_mask])
        
        # Normalize efficiencies
        normalized_efficiencies = efficiencies / median_flat_efficiency
    else:
        print(f"WARNING: Only {np.sum(flat_mask)} flat sections were used to compute efficiency")
        normalized_efficiencies = efficiencies
    
    return normalized_efficiencies


def fit_gap_model(
    gradients: np.ndarray,
    normalized_efficiencies: np.ndarray,
    min_samples_per_bucket: int = 10,
) -> Dict:
    """
    Fit a GAP model using variable width bucketing.
    
    Args:
        gradients: Array of elevation gradients
        normalized_efficiencies: Array of normalized efficiencies
        min_samples_per_bucket: Minimum number of samples per bucket
        
    Returns:
        Dictionary containing:
            - bucket_centers: Array of gradient values at bucket centers
            - bucket_means: Array of mean normalized efficiencies for each bucket
            - bucket_stds: Array of standard deviations for each bucket
            - bucket_counts: Array of sample counts for each bucket
    """
    # Initialize buckets
    bucket_centers = []
    bucket_means = []
    bucket_stds = []
    bucket_counts = []
    
    # Sort data by gradient
    sort_idx = np.argsort(gradients)
    sorted_gradients = gradients[sort_idx]
    sorted_efficiencies = normalized_efficiencies[sort_idx]
    
    # Create buckets with variable width
    current_bucket = []
    current_gradients = []
    
    for i in range(len(sorted_gradients)):
        current_bucket.append(sorted_efficiencies[i])
        current_gradients.append(sorted_gradients[i])
        
        if len(current_bucket) >= min_samples_per_bucket:
            # Calculate bucket statistics
            bucket_centers.append(np.mean(current_gradients))
            bucket_means.append(np.mean(current_bucket))
            bucket_stds.append(np.std(current_bucket))
            bucket_counts.append(len(current_bucket))
            
            # Reset bucket
            current_bucket = []
            current_gradients = []
    
    # Add remaining samples to last bucket
    if current_bucket:
        bucket_centers.append(np.mean(current_gradients))
        bucket_means.append(np.mean(current_bucket))
        bucket_stds.append(np.std(current_bucket))
        bucket_counts.append(len(current_bucket))
    
    return {
        'bucket_centers': np.array(bucket_centers),
        'bucket_means': np.array(bucket_means),
        'bucket_stds': np.array(bucket_stds),
        'bucket_counts': np.array(bucket_counts)
    }


def calculate_gap(
    speed: float,
    elevation_gain: float,
    gap_model: Dict
) -> float:
    """
    Calculate Gradient Adjusted Pace (GAP) using the fitted model.
    
    Args:
        speed: Current speed in km/h
        elevation_gain: Current elevation_gain in m/km
        gap_model: Dictionary containing the fitted GAP model
        
    Returns:
        GAP in km/h
    """
    # Find closest bucket center
    bucket_idx = np.argmin(np.abs(gap_model['bucket_centers'] - elevation_gain))
    
    # Get efficiency factor for this gradient
    efficiency_factor = gap_model['bucket_means'][bucket_idx]
    
    # Calculate GAP
    gap = speed / efficiency_factor
    
    return gap


def plot_gap_curve_centroids(
    gap_model: Dict,
    show_std: bool = True,
    smoothing_parameter: int = 3,
) -> None:
    """
    Plot a smooth curve by connecting the centroids from the fitted model.
    
    Args:
        gap_model: Dictionary containing the fitted GAP model
        show_std: Whether to show standard deviation bounds
        smoothing_parameter: Size of the rolling window for smoothing (must be odd number)
    """
    # Create the plot
    plt.figure(figsize=(12, 6))
    
    # Get the raw data
    centers = gap_model['bucket_centers']
    means = gap_model['bucket_means']
    stds = gap_model['bucket_stds']
    
    # Apply rolling average smoothing
    if smoothing_parameter > 1:
        # Ensure smoothing_parameter is odd
        if smoothing_parameter % 2 == 0:
            smoothing_parameter += 1
        
        # Calculate padding size
        pad_size = (smoothing_parameter - 1) // 2
        
        # Pad the arrays for rolling average
        padded_means = np.pad(means, (pad_size, pad_size), mode='edge')
        padded_stds = np.pad(stds, (pad_size, pad_size), mode='edge')
        
        # Calculate rolling averages
        smoothed_means = np.array([
            np.mean(padded_means[i:i+smoothing_parameter])
            for i in range(len(means))
        ])
        
        smoothed_stds = np.array([
            np.mean(padded_stds[i:i+smoothing_parameter])
            for i in range(len(stds))
        ])
        
        # Plot the smoothed curve
        plt.plot(centers, smoothed_means, 'purple', linewidth=2, label='Smoothed curve')
        
        if show_std:
            plt.fill_between(centers, 
                           smoothed_means - smoothed_stds,
                           smoothed_means + smoothed_stds,
                           alpha=0.2, color='purple', label='±1 std')
    else:
        # Plot the raw curve
        plt.plot(centers, means, 'purple', linewidth=2, label='Centroid curve')
        
        if show_std:
            plt.fill_between(centers, 
                           means - stds,
                           means + stds,
                           alpha=0.2, color='purple', label='±1 std')
    
    # Add labels and title
    plt.xlabel('Elevation Gain (m/km)')
    plt.ylabel('Speed Adjuster (speed/GAP)')
    plt.title('Speed Adjuster vs Elevation Gain (Centroid Curve)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show the plot
    plt.show()