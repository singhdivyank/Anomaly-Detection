import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

from src.consts import *
from ..utils import load_data

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
    data = load_data(RAW_DATA_PATH)
    if data is not None:
        data = handle_missing_values(data)
        data = remove_outliers(data)
        data = normalize_data(data)
        data.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"Processed data saved to {PROCESSED_DATA_PATH}.")
