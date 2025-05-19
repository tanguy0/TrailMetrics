import pandas as pd
import numpy as np
from stravalib import Client
import time
from typing import List, Dict, Tuple


def get_activity_streams(
    client: Client,
    activities: List,
    max_streams: int = None,
    verbose: bool = True
) -> List[Dict]:
    """
    Get streams for provided activities from Strava API.
    
    Args:
        client: Strava API client
        activities: List of activities to process
        max_streams: Maximum number of streams to retrieve (None for all)
        verbose: Whether to print progress information
        
    Returns:
        List of activity stream dictionaries, each containing:
            - time: Array of timestamps in seconds
            - distance: Array of distances in meters
            - altitude: Array of altitudes in meters
            - heartrate: Array of heart rates in bpm
    """
    streams = []
    
    for activity in activities:
        try:
            # Get streams for this activity
            activity_streams = client.get_activity_streams(
                activity.id,
                types=['time', 'distance', 'altitude', 'heartrate'],
                resolution='high'
            )
            
            # Store the streams
            streams.append(activity_streams)
            
            # Check if we've reached the maximum number of streams
            if max_streams and len(streams) >= max_streams:
                break
                
        except Exception as e:
            if verbose:
                print(f"Error getting streams for activity {activity.id}: {str(e)}")
            continue
        
        time.sleep(0.1)
    
    return streams


def filter_outliers(
    speed: np.ndarray,
    elevation_gain: np.ndarray,
    heartrate: np.ndarray,
    sport_types: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out unrealistic values from downsampled data.
    
    Args:
        speed: Array of speeds in km/h
        elevation_gain: Array of elevation gains in m/km
        heartrate: Array of heart rates in bpm
        sport_types: Array of sport type strings
        
    Returns:
        Tuple containing:
            - speed: Filtered array of speeds in km/h
            - elevation_gain: Filtered array of elevation gains in m/km
            - heartrate: Filtered array of heart rates in bpm
            - sport_types: Filtered array of sport type strings
    """
    # Create mask for realistic values
    mask = (
        (elevation_gain >= -350) * (elevation_gain <= 350) *  # Reasonable elevation gain/loss
        (speed >= 3) * (speed <= 22)  # Reasonable speed range (km/h)
    )
    
    return speed[mask], elevation_gain[mask], heartrate[mask], sport_types[mask]


def process_streams(
    streams: List[Dict],
    activity_sport_types: List[str],
    split_min_time: float,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process activity streams into arrays for speed, elevation gain, heart rate, and sport type.
    
    Args:
        streams: List of activity stream dictionaries from get_activity_streams
        activity_sport_types: List of sport types for each activity
        split_min_time: Minimum time in seconds for each split
        verbose: Whether to print progress information
        
    Returns:
        Tuple containing:
            - speeds: Array of speeds in km/h
            - elevation_gains: Array of elevation gains in m/km
            - heartrates: Array of heart rates in bpm
            - sport_types: Array of sport type strings
    """
    speeds = []
    elevation_gains = []
    heartrates = []
    sport_types = []
    
    for i, (activity_stream, sport_type) in enumerate(zip(streams, activity_sport_types)):
        if verbose:
            print(f"\nProcessing streams for activity {i+1}/{len(streams)}")
        
        try:
            # Process single stream
            time, distance, speed, elevation_gain, heartrate = process_single_stream(activity_stream)
            
            # Downsample stream data
            speed, elevation_gain, heartrate = downsample_stream_data(
                time,
                distance,
                speed,
                elevation_gain,
                heartrate,
                split_min_time,
            )
            
            # Store the processed data
            speeds.append(speed)
            elevation_gains.append(elevation_gain)
            heartrates.append(heartrate)
            sport_types.append([sport_type] * len(speed))
            
            if verbose:
                print(f"Successfully processed streams")
                
        except Exception as e:
            if verbose:
                print(f"Error processing streams: {str(e)}")
            continue
    
    # Concatenate all arrays
    all_speeds = np.concatenate(speeds)
    all_elevation_gains = np.concatenate(elevation_gains)
    all_heartrates = np.concatenate(heartrates)
    all_sport_types = np.concatenate(sport_types)
    
    # Filter outliers
    return filter_outliers(
        all_speeds,
        all_elevation_gains,
        all_heartrates,
        all_sport_types
    )


def process_single_stream(
    stream: Dict
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process a single activity stream into arrays of time, distance, speed, elevation gain, and heartrate.
    
    Args:
        stream: Activity stream dictionary containing:
            - time: Array of timestamps in seconds
            - distance: Array of distances in meters
            - altitude: Array of altitudes in meters
            - heartrate: Array of heart rates in bpm
        
    Returns:
        Tuple containing:
            - time: Array of timestamps in seconds
            - distance: Array of distances in meters
            - speed: Array of speeds in km/h
            - elevation_gain: Array of elevation gains in m/km
            - heartrate: Array of heart rates in bpm
    """
    if not stream:
        return None
        
    # Get the data arrays
    time = np.array(stream['time'].data)
    distance = np.array(stream['distance'].data)
    altitude = np.array(stream['altitude'].data)
    heartrate = np.array(stream['heartrate'].data)
    
    # Compute deltas
    delta_dist = np.diff(distance)
    delta_time = np.diff(time)
    
    # Compute instantaneous speed (m/s to km/h)
    speed = (delta_dist / delta_time) * 3.6
    
    # Compute elevation gain (in D+ meters per km of distance ran)
    elevation_gain = np.diff(altitude) / delta_dist * 1000
    
    # Align all array lengths
    time = time[1:]
    distance = distance[1:]
    heartrate = heartrate[1:]
    
    return time, distance, speed, elevation_gain, heartrate


def downsample_stream_data(
    time: np.ndarray,
    distance: np.ndarray,
    speed: np.ndarray,
    elevation_gain: np.ndarray,
    heartrate: np.ndarray,
    split_min_time: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Downsample stream data into fixed time splits, excluding splits with both positive and negative elevation changes.
    
    Args:
        time: Array of timestamps in seconds
        distance: Array of distances in meters
        speed: Array of speeds in km/h
        elevation_gain: Array of elevation gains in m/km
        heartrate: Array of heart rates in bpm
        split_min_time: Minimum time in seconds for each split
        
    Returns:
        Tuple containing:
            - speed: Array of average speeds in km/h for each split
            - elevation_gain: Array of average elevation gains in m/km for each split
            - heartrate: Array of average heart rates in bpm for each split
    """
    # Cut first 15 minutes to avoid warm-up bias
    idx_cut = (time < 60 * 15).sum()
    if idx_cut > 0:
        time = time[idx_cut:]
        distance = distance[idx_cut:]
        speed = speed[idx_cut:]
        elevation_gain = elevation_gain[idx_cut:]
        heartrate = heartrate[idx_cut:]
    
    # Find indices for aggregation
    cuts = []
    current_idx = 0
    for i in range(1, len(time)):
        if time[i] - time[current_idx] >= split_min_time:
            cuts.append(i)
            current_idx = i
    
    # Downsample data
    agg_speed = []
    agg_elevation_gain = []
    agg_heartrate = []
    
    current_idx = 0
    for cut_idx in cuts:
        # Get data for this split
        split_speed = speed[current_idx:cut_idx]
        split_elevation = elevation_gain[current_idx:cut_idx]
        split_heartrate = heartrate[current_idx:cut_idx]
        
        # Check if split has both positive and negative elevation changes
        if not (np.any(split_elevation > 0) and np.any(split_elevation < 0)):
            agg_speed.append(split_speed.mean())
            agg_elevation_gain.append(split_elevation.mean())
            agg_heartrate.append(split_heartrate.mean())
        
        current_idx = cut_idx
    
    return (
        np.array(agg_speed),
        np.array(agg_elevation_gain),
        np.array(agg_heartrate)
    )


def prepare_ml_calibration_dataset(
    speeds: np.ndarray,
    elevation_gains: np.ndarray,
    heartrates: np.ndarray,
    flat_elevation_gain_range: Tuple[float, float] = 10.0,
    hr_tolerance: float = 3.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset for machine learning model calibration by matching points with similar heart rates.
    
    Args:
        speeds: Array of speeds in km/h
        elevation_gains: Array of elevation gains in m/km
        heartrates: Array of heart rates in bpm
        flat_elevation_gain_range: Tuple of (min, max) elevation gain in m/km to consider as flat
        hr_tolerance: Maximum difference in heart rate to consider points as similar
        
    Returns:
        Tuple containing:
            - X: Array of shape (n_samples, 3) with [speed, elevation_gain, heartrate]
            - y: Array of average speeds from matching flat points
    """
    # Find points with nearly flat elevation
    flat_mask = (elevation_gains > flat_elevation_gain_range[0]) * (elevation_gains < flat_elevation_gain_range[1])
    
    # Initialize lists to store matched points
    X_list = []
    y_list = []
    
    # For each point
    for i in range(len(speeds)):
        # Find flat points with similar heart rate
        hr_diff = np.abs(heartrates[flat_mask] - heartrates[i])
        similar_hr_mask = hr_diff <= hr_tolerance
        
        if np.any(similar_hr_mask):
            # Get the average speed of matching flat points
            matching_speeds = speeds[flat_mask][similar_hr_mask]

            # Add the point to our dataset to match all the flat speeds
            for matching_speed in matching_speeds:
                X_list.append([speeds[i], elevation_gains[i], heartrates[i]])
                y_list.append(matching_speed)
    
    return np.array(X_list), np.array(y_list)