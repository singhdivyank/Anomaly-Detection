import pandas as pd

def add_time_features(data, timestamp_col='timestamp'):
    """
    Extract time-based features from a timestamp column.

    This function adds the following columns:
      - hour: Hour of the day
      - day_of_week: Day of the week (0=Monday, 6=Sunday)
      - month: Month of the year
      - is_weekend: Boolean flag indicating if the day is a weekend

    Args:
        data (pd.DataFrame): DataFrame containing a timestamp column.
        timestamp_col (str): Name of the timestamp column.
    
    Returns:
        pd.DataFrame: DataFrame with new time features added.
    """
    # Convert the timestamp column to datetime
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    
    # Extract time-based features
    data['hour'] = data[timestamp_col].dt.hour
    data['day_of_week'] = data[timestamp_col].dt.dayofweek
    data['month'] = data[timestamp_col].dt.month
    data['is_weekend'] = data['day_of_week'].apply(lambda x: x >= 5)
    return data

def categorize_tariff(data, tariff_col='tariff'):
    """
    Categorize tariff values into bins.

    This function converts a continuous tariff column into categorical bins.
    The bin thresholds and labels can be adjusted according to your data distribution.
    
    Args:
        data (pd.DataFrame): DataFrame containing a tariff column.
        tariff_col (str): Name of the tariff column.
    
    Returns:
        pd.DataFrame: DataFrame with a new column 'tariff_category' added.
    """
    # Example bins and labels; adjust as needed based on your dataset's tariff range.
    bins = [0, 0.1, 0.2, 0.3, 1.0]
    labels = ['Very Low', 'Low', 'Medium', 'High']
    if tariff_col in data.columns:
        data['tariff_category'] = pd.cut(data[tariff_col], bins=bins, labels=labels, include_lowest=True)
    else:
        print(f"Column '{tariff_col}' not found in data.")
    return data

if __name__ == "__main__":
    # Example usage:
    processed_file_path = "../data/processed/smart_meter_data_processed.csv"
    output_file_path = "../data/processed/smart_meter_data_features.csv"
    
    # Load the processed data
    data = pd.read_csv(processed_file_path)
    
    # Add time-based features from the timestamp column
    data = add_time_features(data, 'timestamp')
    
    # Categorize the tariff values (ensure your dataset has a 'tariff' column)
    data = categorize_tariff(data, 'tariff')
    
    # Save the dataset with engineered features
    data.to_csv(output_file_path, index=False)
    print(f"Feature engineered data saved to {output_file_path}.")
