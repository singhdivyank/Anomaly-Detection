import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

def train_isolation_forest(data, features, contamination='auto', random_state=42, model_path="models/isolation_forest_model.pkl"):
    if os.path.exists(model_path):
        print(f"Loading existing Isolation Forest model from {model_path}")
        model = joblib.load(model_path)
        return model

    print("Training new Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=random_state, n_jobs=-1)
    model.fit(data[features])
    joblib.dump(model, model_path)
    print(f"Isolation Forest model saved to {model_path}")
    return model

def predict_anomalies(model, data, features):
    data['anomaly_score'] = model.decision_function(data[features])
    data['anomaly'] = (model.predict(data[features]) == -1).astype(int)
    return data

if __name__ == "__main__":
    data_path = "data/processed/smart_meter_data_features.csv"
    model_output_path = "models/isolation_forest_model.pkl"
    output_data_path = "data/processed/smart_meter_data_anomalies_if.csv"

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please run preprocessing and feature engineering first.")
        exit()

    numeric_features = data.select_dtypes(include=np.number).columns.tolist()
    features_to_exclude = ['cluster', 'anomaly_score', 'anomaly']
    features = [col for col in numeric_features if col not in features_to_exclude]

    if not features:
        print("Error: No suitable numeric features found for training.")
        exit()

    print("Using features for Isolation Forest:", features)

    model = train_isolation_forest(data, features, model_path=model_output_path)
    data_with_anomalies = predict_anomalies(model, data.copy(), features)

    anomaly_count = data_with_anomalies['anomaly'].sum()
    print(f"Number of anomalies detected: {anomaly_count} ({anomaly_count / len(data_with_anomalies):.2%})")

    data_with_anomalies.to_csv(output_data_path, index=False)
    print(f"Data with anomaly predictions saved to {output_data_path}")