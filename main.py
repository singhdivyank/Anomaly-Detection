import joblib

from src.data_preprocessing import load_data
from src.feature_engineering import add_time_features, categorize_tariff
from src.model_random_forest import train_random_forest
from src.model_neural_network import train_neural_network
from src.consts import *
from utils import *

def main():
    # upload raw data
    print("Loading raw data...")
    data = load_data(RAW_DATA_PATH)
    if data is None:
        return

    # perform initial steps
    preprocessed_data = preprocess(data)
    preprocessed_data.to_csv(PROCESSED_DATA_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_DATA_PATH}")
    
    # Feature Engineering
    print("Adding time features...")
    feature_data = add_time_features(preprocessed_data, timestamp_col="timestamp")
    # categorize tariff
    print("Categorizing tariff...")
    tariff_data = categorize_tariff(feature_data, tariff_col="tariff")
    tariff_data.to_csv(FEATURES_DATA_PATH, index=False)
    print(f"Feature engineered data saved to {FEATURES_DATA_PATH}")
    
    # perform clustering
    perform_clustering(data=tariff_data)
    # supervised learning
    target = "anomaly"
    if target not in data.columns:
        print(f"Target column {target} not found. Skipping supervised training.")
    else:
        # Exclude columns that aren't features
        features = [col for col in data.columns if col not in ["timestamp", target, "cluster"]]
        # Train Random Forest Classifier
        print("Training Random Forest model...")
        rf_model = train_random_forest(data, features, target)
        joblib.dump(rf_model, RF_PATH)
        print(f"Random Forest model saved to {RF_PATH}")
        # Train Neural Network Classifier
        print("Training Neural Network model...")
        nn_model, _, _, _, scaler = train_neural_network(data, features, target, epochs=20)
        nn_model.save(NN_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print(f"Neural Network model saved to {NN_PATH}")
        print(f"Scaler saved to {SCALER_PATH}")


if __name__ == "__main__":
    main()
