import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model

def build_model(input_dim):
    """
    Build a simple feedforward neural network model.
    
    Args:
        input_dim (int): Number of features.
        
    Returns:
        model: Compiled Keras model.
    """
    model = Sequential()
    model.add(Dense(64, activation='relu', input_dim=input_dim))
    model.add(Dropout(0.3))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification (normal vs. anomaly)
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_neural_network(data, feature_columns, target, test_size=0.2, random_state=42, epochs=50, batch_size=32):
    """
    Train a neural network model on the provided dataset.
    
    Args:
        data (pd.DataFrame): DataFrame containing features and target.
        feature_columns (list): List of feature column names.
        target (str): Target column name (e.g., 'anomaly').
        test_size (float): Fraction of data to be used as test set.
        random_state (int): Random seed for reproducibility.
        epochs (int): Number of epochs for training.
        batch_size (int): Batch size for training.
        
    Returns:
        model: Trained Keras model.
        history: Training history.
        X_test, y_test: Test data for evaluation.
    """
    X = data[feature_columns].values
    y = data[target].values
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # Build model
    model = build_model(input_dim=X_train.shape[1])
    
    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    history = model.fit(
        X_train, y_train, 
        validation_split=0.2, 
        epochs=epochs, 
        batch_size=batch_size, 
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate the model on the test set
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.3f}, Test Accuracy: {accuracy:.3f}")
    
    return model, history, X_test, y_test, scaler

if __name__ == "__main__":
    # Paths for data and model output
    data_path = "../data/processed/smart_meter_data_features.csv"
    model_output_path = "../models/nn_model.h5"
    scaler_output_path = "../models/nn_scaler.pkl"
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Check if the target label for abnormal responses exists
    target = 'anomaly'  # This column should indicate normal (0) vs. abnormal (1)
    if target not in data.columns:
        print(f"Target column '{target}' not found. Please ensure your dataset includes anomaly labels.")
    else:
        # Select feature columns (exclude non-feature columns)
        feature_columns = [col for col in data.columns if col not in ['timestamp', target, 'cluster']]
        if not feature_columns:
            print("No feature columns available for training.")
        else:
            # Train the neural network model
            model, history, X_test, y_test, scaler = train_neural_network(data, feature_columns, target)
            
            # Save the trained model
            model.save(model_output_path)
            print(f"Neural Network model saved to {model_output_path}.")
            
            # Optionally, save the scaler for future use (requires joblib)
            import joblib
            joblib.dump(scaler, scaler_output_path)
            print(f"Scaler saved to {scaler_output_path}.")
