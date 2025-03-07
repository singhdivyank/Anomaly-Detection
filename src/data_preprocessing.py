import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def load_data(file_path):
    """
    Load data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(data):
    """
    Handle missing values in the DataFrame.
    
    This function fills missing numerical values with the column mean.
    
    Args:
        data (pd.DataFrame): DataFrame with potential missing values.
    
    Returns:
        pd.DataFrame: DataFrame with missing values handled.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean_value = data[col].mean()
        data[col].fillna(mean_value, inplace=True)
    return data

def remove_outliers(data, z_threshold=3):
    """
    Remove outliers from numerical columns using a z-score threshold.
    
    Args:
        data (pd.DataFrame): DataFrame to process.
        z_threshold (float): The z-score threshold to identify outliers.
    
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
    """
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    mask = (z_scores < z_threshold).all(axis=1)
    return data[mask]

def normalize_data(data):
    """
    Normalize numerical columns using standard scaling.
    
    Args:
        data (pd.DataFrame): DataFrame with numerical data.
    
    Returns:
        pd.DataFrame: DataFrame with normalized numerical columns.
    """
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

if __name__ == "__main__":
    # Example usage:
    # Update the file paths as needed for your project structure.
    raw_file_path = "../data/raw/smart_meter_data.csv"  # Replace with your actual data file
    processed_file_path = "../data/processed/smart_meter_data_processed.csv"
    
    data = load_data(raw_file_path)
    if data is not None:
        data = handle_missing_values(data)
        data = remove_outliers(data)
        data = normalize_data(data)
        
        # Save the processed data for later use
        data.to_csv(processed_file_path, index=False)
        print(f"Processed data saved to {processed_file_path}.")
