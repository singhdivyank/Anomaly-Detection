import os
import pandas as pd
import numpy as np
import joblib

import lightgbm as lgb

from sklearn.model_selection import train_test_split

from .consts import *
from ..utils import *

def train_lightgbm(data, features, target):
    if os.path.exists(LIGHTGBM_PATH):
        print(f"Loading existing LightGBM model from {LIGHTGBM_PATH}")
        model = joblib.load(LIGHTGBM_PATH)
        return model

    # segregate features and target
    X, y = data[features], data[target]
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training new LightGBM model...")
    model = lgb.LGBMClassifier(random_state=42, n_jobs=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss', callbacks=[lgb.early_stopping(10, verbose=False)])
    joblib.dump(model, LIGHTGBM_PATH)
    print(f"LightGBM model saved to {LIGHTGBM_PATH}")
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred)

if __name__ == "__main__":
    data = pd.read_csv(ANOMALY_FILE_PATH)
    if 'anomaly' not in data.columns:
        print("Target column 'anomaly' not found.")
        exit()

    numeric_features = data.select_dtypes(include=np.number).columns.tolist()
    features = [col for col in numeric_features if col not in ['cluster', 'anomaly_score', 'anomaly']]
    if not features:
        print("No suitable numeric features found for training.")
        exit()

    print("Using features for LightGBM:", features)
    train_lightgbm(data, features, 'anomaly')
