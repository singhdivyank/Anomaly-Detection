import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, skipinitialspace=True)
        data.columns = data.columns.str.strip()
        # Convert energy column
        if 'KWH/hh (per half hour)' in data.columns:
            data['KWH/hh (per half hour)'] = pd.to_numeric(data['KWH/hh (per half hour)'], errors='coerce')
        # Convert tariff column if exists
        if 'tariff' in data.columns:
            data['tariff'] = pd.to_numeric(data['tariff'], errors='coerce')
        print(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def handle_missing_values(data):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        mean_value = data[col].mean()
        data[col].fillna(mean_value, inplace=True)
    return data

def remove_outliers(data, z_threshold=3):
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    z_scores = np.abs(stats.zscore(data[numeric_cols]))
    mask = (z_scores < z_threshold).all(axis=1)
    return data[mask]

def normalize_data(data):
    scaler = StandardScaler()
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data

if __name__ == "__main__":
    raw_file_path = "data/raw/smart_meter_data_dtou.csv"
    processed_file_path = "data/processed/smart_meter_data_processed.csv"

    data = load_data(raw_file_path)
    if data is not None:
        data = handle_missing_values(data)
        data = remove_outliers(data)
        data = normalize_data(data)

        data.to_csv(processed_file_path, index=False)
        print(f"Processed data saved to {processed_file_path}.")
