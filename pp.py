import pandas as pd
import numpy as np
import json

def load_dataset(filepath, features):
    """Load and clean dataset by dropping missing sensor values."""
    print(f"Loading '{filepath}' for mathematical modeling...")
    df = pd.read_csv(filepath)
    df_clean = df.dropna(subset=features).copy()
    return df_clean

def compute_spatial_kinematics(df):
    """
    Model 1: Spatial Kinematics & Gravitational Tilt Angles
    Computes Signal Magnitude Vector (SMV), Pitch, and Roll Eulerian angles.
    """
    # Signal Magnitude Vector (SMV)
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['gyro_mag'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)
    
    # Pitch (theta) and Roll (phi) in degrees
    df['pitch_deg'] = np.degrees(np.arctan2(df['acc_x'], np.sqrt(df['acc_y']**2 + df['acc_z']**2)))
    df['roll_deg'] = np.degrees(np.arctan2(df['acc_y'], np.sqrt(df['acc_x']**2 + df['acc_z']**2)))
    return df

def analyze_activity_kinetics(df_act):
    """
    Model 2: Kinetic Intensity & Signal Variance
    Calculates mean magnitude, variance of acceleration, and angular energy.
    """
    print("\n" + "="*70)
    print(" MODEL 1 & 2: DYNAMIC KINETICS & SPATIAL ORIENTATION (activity.csv)")
    print("="*70)
    
    df_act = compute_spatial_kinematics(df_act)
    
    # Group by activity and calculate kinematic summary statistics
    stats = df_act.groupby('activity').agg({
        'acc_mag': ['mean', 'var'],
        'gyro_mag': ['mean', 'var'],
        'pitch_deg': 'mean',
        'roll_deg': 'mean',
        'barometer': 'mean'
    }).reset_index()
    
    # Flatten multi-level columns
    stats.columns = ['Activity', 'Acc_Mag_Mean', 'Acc_Mag_Var', 'Gyro_Mag_Mean', 'Gyro_Mag_Var', 'Pitch_Mean_Deg', 'Roll_Mean_Deg', 'Baro_Mean_hPa']
    
    print(stats.to_string(index=False))
    return stats

def analyze_sitting_postures(df_sit):
    """
    Model 1 Applied: Ergonomic Sitting Gravitational Vector Analysis
    """
    print("\n" + "="*70)
    print(" MODEL 1: MICRO-ERGONOMIC SITTING GRAVITATIONAL VECTORS (sitting.csv)")
    print("="*70)
    
    df_sit = compute_spatial_kinematics(df_sit)
    
    sit_stats = df_sit.groupby('posture').agg({
        'acc_x': 'mean',
        'acc_y': 'mean',
        'acc_z': 'mean',
        'acc_mag': 'mean',
        'pitch_deg': 'mean',
        'roll_deg': 'mean'
    }).reset_index()
    
    sit_stats.columns = ['Sitting Posture', 'Acc_X_Mean', 'Acc_Y_Mean', 'Acc_Z_Mean', 'Acc_Mag_Mean', 'Pitch_Deg', 'Roll_Deg']
    print(sit_stats.to_string(index=False))
    return sit_stats

def analyze_sleeping_postures(df_sleep):
    """
    Model 1 Applied: Nocturnal Sleep Position Spatial Orientations
    """
    print("\n" + "="*70)
    print(" MODEL 1: NOCTURNAL SLEEP POSITION SPATIAL ORIENTATIONS (sleeping.csv)")
    print("="*70)
    
    df_sleep = compute_spatial_kinematics(df_sleep)
    
    sleep_stats = df_sleep.groupby('posture').agg({
        'acc_x': 'mean',
        'acc_y': 'mean',
        'acc_z': 'mean',
        'pitch_deg': 'mean',
        'roll_deg': 'mean'
    }).reset_index()
    
    sleep_stats.columns = ['Sleep Posture', 'Acc_X_Mean', 'Acc_Y_Mean', 'Acc_Z_Mean', 'Pitch_Deg', 'Roll_Deg']
    print(sleep_stats.to_string(index=False))
    return sleep_stats

def analyze_barometric_elevation(df_act):
    """
    Model 3: Hypsometric Barometric Elevation Model
    Calculates height delta (dh) from pressure differential (dP).
    """
    print("\n" + "="*70)
    print(" MODEL 3: HYPSOMETRIC BAROMETRIC ELEVATION DIFFERENTIALS")
    print("="*70)
    
    sit_baro = df_act[df_act['activity'] == 'Sitting']['barometer'].mean()
    stand_baro = df_act[df_act['activity'] == 'Standing']['barometer'].mean()
    
    # Delta Pressure in hPa and Pa
    dP_hPa = sit_baro - stand_baro
    dP_Pa = dP_hPa * 100.0
    
    # Sensitivity constant: ~8.43 cm per Pa shift
    dh_cm = dP_Pa * 0.0843
    
    print(f" Mean Sitting Barometric Pressure:  {sit_baro:.4f} hPa")
    print(f" Mean Standing Barometric Pressure: {stand_baro:.4f} hPa")
    print(f" Pressure Differential (dP):       {dP_hPa:.4f} hPa ({dP_Pa:.2f} Pa)")
    print(f" Calculated Vertical Height Shift:  {dh_cm:.2f} cm")

def main():
    features_full = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'barometer']
    features_kinematic = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    
    # Load raw data
    df_act = load_dataset('activity.csv', features_full)
    df_sit = load_dataset('sitting.csv', features_full)
    df_sleep = load_dataset('sleeping.csv', features_kinematic)
    
    # Run mathematical modeling functions
    analyze_activity_kinetics(df_act)
    analyze_sitting_postures(df_sit)
    analyze_sleeping_postures(df_sleep)
    analyze_barometric_elevation(df_act)

if __name__ == '__main__':
    main()