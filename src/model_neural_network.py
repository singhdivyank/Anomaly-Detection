import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import joblib
import os

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

def train_neural_network(data, feature_columns, target, test_size=0.2, random_state=42, epochs=10, batch_size=32, model_path="models/nn_model.h5", scaler_path="models/nn_scaler.pkl"):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"Loading existing Neural Network model from {model_path} and scaler from {scaler_path}")
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        X = data[feature_columns].values
        y = data[target].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        X_test = scaler.transform(X_test)
        return model, None, X_test, y_test, scaler

    print("Training new Neural Network model...")
    X = data[feature_columns].values
    y = data[target].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = build_model(input_dim=X_train.shape[1])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=1
    )

    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.3f}")

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Neural Network model saved to {model_path}")
    print(f"Scaler saved to {scaler_path}")

    return model, history, X_test, y_test, scaler

if __name__ == "__main__":
    data_path = "data/processed/smart_meter_data_anomalies_if.csv"
    model_output_path = "models/nn_model.h5"
    scaler_output_path = "models/nn_scaler.pkl"

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
    feature_columns = [col for col in numeric_features if col not in features_to_exclude]

    if not feature_columns:
        print("No suitable numeric features found for training.")
        exit()

    print("Using features for Neural Network:", feature_columns)

    model, history, X_test, y_test, scaler = train_neural_network(
        data, feature_columns, target, model_path=model_output_path, scaler_path=scaler_output_path
    )

    if history:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,6))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Neural Network Training Loss')
        plt.legend()
        plt.show()
