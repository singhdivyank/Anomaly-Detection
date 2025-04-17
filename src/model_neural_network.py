import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.consts import *
from src.utils import plot_loss

Sequential = tf.keras.models.Sequential
load_model = tf.keras.models.load_model
Dense = tf.keras.layers.Dense
Dropout = tf.keras.layers.Dropout
EarlyStopping = tf.keras.callbacks.EarlyStopping

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_existing(target, data):
    if os.path.exists(NN_PATH) and os.path.exists(SCALER_PATH):
        print(f"Loading existing Neural Network model from {NN_PATH} and scaler from {SCALER_PATH}")
        model = load_model(NN_PATH)
        scaler = joblib.load(SCALER_PATH)
        X, y = data[feature_columns].values, data[target].values
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_test = scaler.transform(X_test)
        return model, None, X_test, y_test, scaler

def train_neural_network(data, feature_columns, target):
    
    scaler = StandardScaler()
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    model, history, X_test, y_test, _ = load_existing(target)
    print("Training new Neural Network model...")
    X, y = data[feature_columns].values, data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)

    model = build_model(input_dim=X_train.shape[1])
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=10,
        batch_size=32,
        callbacks=[early_stop],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.3f}")

    model.save(NN_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Neural Network model saved to {NN_PATH}\nScaler saved to {SCALER_PATH}")
    return model, history, X_test, y_test, scaler

if __name__ == "__main__":
    data = pd.read_csv(ANOMALY_FILE_PATH)
    if 'anomaly' not in data.columns:
        print(f"Target column 'anomaly' not found.")
        exit()

    numeric_features = data.select_dtypes(include=np.number).columns.tolist()
    feature_columns = [col for col in numeric_features if col not in ['cluster', 'anomaly_score', 'anomaly']]
    if not feature_columns:
        print("No suitable numeric features found for training.")
        exit()

    print("Using features for Neural Network:", feature_columns)
    model, history, X_test, y_test, scaler = train_neural_network(data, feature_columns, 'anomaly', model_path=NN_PATH, scaler_path=SCALER_PATH)
    if history:
        plot_loss(history=history)
