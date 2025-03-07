import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

def train_random_forest(data, features, target, test_size=0.2, random_state=42):
    """
    Train a Random Forest classifier on the provided dataset.

    Args:
        data (pd.DataFrame): DataFrame containing the features and target.
        features (list): List of feature column names.
        target (str): Target column name (e.g., 'anomaly').
        test_size (float): Fraction of data to be used as the test set.
        random_state (int): Seed for reproducibility.

    Returns:
        RandomForestClassifier: The trained Random Forest model.
    """
    # Split the dataset into training and testing sets
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Initialize and train the Random Forest model
    model = RandomForestClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate model performance
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)
    
    return model

if __name__ == "__main__":
    # Path to the feature-engineered data
    data_path = "../data/processed/smart_meter_data_features.csv"
    model_output_path = "../models/rf_model.pkl"
    
    # Load the dataset
    data = pd.read_csv(data_path)
    
    # Check if the target label for abnormal responses exists in the dataset
    target = 'anomaly'  # This column should indicate normal (0) vs abnormal (1) responses
    if target not in data.columns:
        print(f"Target column '{target}' not found. Please ensure your dataset includes anomaly labels.")
    else:
        # Select feature columns (excluding non-feature columns like timestamps and any pre-existing labels)
        feature_columns = [col for col in data.columns if col not in ['timestamp', target, 'cluster']]
        
        if not feature_columns:
            print("No feature columns available for training.")
        else:
            # Train the Random Forest model
            model = train_random_forest(data, feature_columns, target)
            
            # Save the trained model for later use
            joblib.dump(model, model_output_path)
            print(f"Random Forest model saved to {model_output_path}.")
