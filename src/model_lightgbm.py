import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import lightgbm as lgb
import joblib
import os

def train_lightgbm(data, features, target, test_size=0.2, random_state=42, model_path="models/lightgbm_model.pkl"):
    if os.path.exists(model_path):
        print(f"Loading existing LightGBM model from {model_path}")
        model = joblib.load(model_path)
        return model

    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Training new LightGBM model...")
    model = lgb.LGBMClassifier(random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train,
              eval_set=[(X_test, y_test)],
              eval_metric='logloss',
              callbacks=[lgb.early_stopping(10, verbose=False)])

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    joblib.dump(model, model_path)
    print(f"LightGBM model saved to {model_path}")

    return model

if __name__ == "__main__":
    data_path = "data/processed/smart_meter_data_anomalies_if.csv"
    model_output_path = "models/lightgbm_model.pkl"

    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}. Please run the Isolation Forest script first.")
        exit()

    target = 'anomaly'
    if target not in data.columns:
        print(f"Target column '{target}' not found.")
        exit()

    numeric_features = data.select_dtypes(include=np.number).columns.tolist()
    features_to_exclude = ['cluster', 'anomaly_score', 'anomaly']
    features = [col for col in numeric_features if col not in features_to_exclude]

    if not features:
        print("No suitable numeric features found for training.")
        exit()

    print("Using features for LightGBM:", features)

    model = train_lightgbm(data, features, target, model_path=model_output_path)