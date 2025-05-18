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


def calculate_efficiency_model_gap(
    speed: float,
    elevation_gain: float,
    efficiency_gap_model: Dict
) -> float:
    """
    Calculate Gradient Adjusted Pace (GAP) using the fitted efficiency model.
    
    Args:
        speed: Current speed in km/h
        elevation_gain: Current elevation_gain in m/km
        gap_model: Dictionary containing the fitted GAP model
        
    Returns:
        GAP in km/h
    """
    # Find closest bucket center
    bucket_idx = np.argmin(np.abs(efficiency_gap_model['bucket_centers'] - elevation_gain))
    
    # Get efficiency factor for this gradient
    efficiency_factor = efficiency_gap_model['bucket_means'][bucket_idx]
    
    # Calculate GAP
    gap = speed * efficiency_factor
    
    return gap


def plot_gap_curves(
    gap_curves: Dict
) -> plt.Figure:
    """
    Plot multiple GAP curves on the same figure.
    
    Args:
        gap_curves: Dictionary of GAP curves, where each curve is a dictionary containing:
            - bin_centers: Array of elevation gain values
            - means: Array of mean speed adjusters
            - stds: Array of standard deviations
            - color: Color for the curve and its standard deviation bounds
            
    Returns:
        Figure object with the plotted curves
    """
    fig = plt.figure(figsize=(12, 6))
    
    for gap_curve_name, gap_curve in gap_curves.items():
        plt.plot(
            gap_curve['bin_centers'], 
            gap_curve['means'], 
            gap_curve['color'], 
            linewidth=2, 
            label=f'{gap_curve_name} GAP curve'
        )
        plt.fill_between(
            gap_curve['bin_centers'], 
            gap_curve['means'] - gap_curve['stds'],
            gap_curve['means'] + gap_curve['stds'],
            alpha=0.2, 
            color=gap_curve['color'], 
            label='±1 std'
        )

    plt.xlabel('Elevation Gain (m/km)')
    plt.ylabel('Speed Adjuster (GAP/speed)')
    plt.title('GAP Curve(s) and standard deviation(s)')
    plt.grid(True, alpha=0.3)
    plt.legend()

    return fig


def get_efficiency_model_gap_curve(
    gap_model: Dict,
    smoothing_parameter: int = 3,
) -> Dict:
    """
    Create a smooth curve by connecting the centroids from the fitted model.
    
    Args:
        gap_model: Dictionary containing the fitted GAP model with:
            - bucket_centers: Array of gradient values
            - bucket_means: Array of mean normalized efficiencies
            - bucket_stds: Array of standard deviations
            - bucket_counts: Array of sample counts
        smoothing_parameter: Size of the rolling window for smoothing (must be odd number)
        
    Returns:
        Dictionary with curve data:
            - bin_centers: Array of gradient values
            - means: Array of mean normalized efficiencies
            - stds: Array of standard deviations
            - counts: Array of sample counts
    """
    # Create the plot
    fig = plt.figure(figsize=(12, 6))
    
    # Get the raw data
    centers = gap_model['bucket_centers']
    means = gap_model['bucket_means']
    stds = gap_model['bucket_stds']
    counts = gap_model['bucket_counts']
    
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
    
    # Create curve data dictionary
    curve_data = {
        'bin_centers': centers,
        'means': smoothed_means if smoothing_parameter > 1 else means,
        'stds': smoothed_stds if smoothing_parameter > 1 else stds,
        'counts': counts
    }
    
    return curve_data


def get_xgboost_gap_curve(
    model,
    X: np.ndarray,
    bin_width: float = 1.0,
    heartrate_range: Tuple[float, float] = None,
) -> Dict:
    """
    Create the GAP curve for the XGBoost model.
    
    Args:
        model: Trained XGBoost model
        X: Data points with shape (n_samples, 3) containing [speed, elevation_gain, heartrate]
        bin_width: Width of elevation gain bins in m/km
        heartrate_range: Optional tuple of (min_hr, max_hr) to filter data points
        
    Returns:
        Dictionary with curve data:
            - bin_centers: Array of elevation gain values
            - means: Array of mean speed adjusters
            - stds: Array of standard deviations
            - counts: Array of point counts per bin
    """
    # Filter data points by heart rate if range is provided
    if heartrate_range is not None:
        hr_mask = (X[:, 2] >= heartrate_range[0]) & (X[:, 2] <= heartrate_range[1])
        X = X[hr_mask]
        if len(X) == 0:
            raise ValueError(f"No data points found in heart rate range {heartrate_range}")
    
    # Get predictions
    gaps = model.predict(X)
    
    # Calculate speed adjusters
    speed_adjusters = gaps / X[:, 0]
    
    # Create bins for elevation gains
    min_elev = np.floor(np.min(X[:, 1]) / bin_width) * bin_width
    max_elev = np.ceil(np.max(X[:, 1]) / bin_width) * bin_width
    bin_edges = np.arange(min_elev, max_elev + bin_width, bin_width)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Calculate average and std of speed adjuster per bin
    avg_speed_adjusters = []
    std_speed_adjusters = []
    bin_counts = []
    for i in range(len(bin_edges) - 1):
        mask = (X[:, 1] >= bin_edges[i]) & (X[:, 1] < bin_edges[i + 1])
        if np.any(mask):
            avg_speed_adjusters.append(np.mean(speed_adjusters[mask]))
            std_speed_adjusters.append(np.std(speed_adjusters[mask]))
            bin_counts.append(np.sum(mask))
        else:
            avg_speed_adjusters.append(np.nan)
            std_speed_adjusters.append(np.nan)
            bin_counts.append(0)
    
    # Convert to numpy arrays
    avg_speed_adjusters = np.array(avg_speed_adjusters)
    std_speed_adjusters = np.array(std_speed_adjusters)
    bin_counts = np.array(bin_counts)
    
    # Create curve data dictionary
    curve_data = {
        'bin_centers': bin_centers,
        'means': avg_speed_adjusters,
        'stds': std_speed_adjusters,
        'counts': bin_counts
    }
    
    return curve_data