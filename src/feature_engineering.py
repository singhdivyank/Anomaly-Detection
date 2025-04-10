import pandas as pd

def add_time_features(data, timestamp_col='DateTime'):
    data[timestamp_col] = pd.to_datetime(data[timestamp_col])
    data['hour'] = data[timestamp_col].dt.hour
    data['day_of_week'] = data[timestamp_col].dt.dayofweek
    data['month'] = data[timestamp_col].dt.month
    data['is_weekend'] = data['day_of_week'].apply(lambda x: x >= 5)
    return data

def categorize_tariff(data, tariff_col='tariff'):
    bins = [0, 0.1, 0.2, 0.3, 1.0]
    labels = ['Very Low', 'Low', 'Medium', 'High']
    if tariff_col in data.columns:
        data['tariff_category'] = pd.cut(data[tariff_col], bins=bins, labels=labels, include_lowest=True)
    else:
        print(f"Column '{tariff_col}' not found in data.")
    return data

if __name__ == "__main__":
    input_path = "data/processed/smart_meter_data_processed.csv"
    output_path = "data/processed/smart_meter_data_features.csv"

    data = pd.read_csv(input_path)

    data = add_time_features(data, 'DateTime')
    data = categorize_tariff(data, 'tariff')

    data.to_csv(output_path, index=False)
    print(f"Feature engineered data saved to {output_path}.")
