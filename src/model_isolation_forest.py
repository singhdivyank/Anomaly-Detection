import pandas as pd
import numpy as np
import joblib
import os

from sklearn.ensemble import IsolationForest

from .consts import *

def train_isolation_forest(data, features):
    if os.path.exists(ISOLATION_FOREST_PATH):
        print(f"Loading existing Isolation Forest model from {ISOLATION_FOREST_PATH}")
        model = joblib.load(ISOLATION_FOREST_PATH)
        return model

    print("Training new Isolation Forest model...")
    model = IsolationForest(n_estimators=100, contamination='auto', random_state=42, n_jobs=-1)
    model.fit(data[features])
    joblib.dump(model, ISOLATION_FOREST_PATH)
    print(f"Isolation Forest model saved to {ISOLATION_FOREST_PATH}")
    return model

def predict_anomalies(model, data, features):
    data['anomaly_score'] = model.decision_function(data[features])
    data['anomaly'] = (model.predict(data[features]) == -1).astype(int)
    return data

if __name__ == "__main__":
    data = pd.read_csv(FEATURES_DATA_PATH)
    numeric_features = data.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric_features if col not in ['cluster', 'anomaly_score', 'anomaly']]
    if not features:
        print("Error: No suitable numeric features found for training.")
        exit()

    print("Using features for Isolation Forest:", features)
    model = train_isolation_forest(data, features)
    anomalies = predict_anomalies(model, data.copy(), features)
    anomaly_count = anomalies['anomaly'].sum()
    print(f"Number of anomalies detected: {anomaly_count} ({anomaly_count / len(anomalies):.2%})")
    anomalies.to_csv(ANOMALY_FILE_PATH, index=False)
    print(f"Data with anomaly predictions saved to {ANOMALY_FILE_PATH}")
