import pandas as pd
import joblib
from src.data_preprocessing import load_data, handle_missing_values, remove_outliers, normalize_data
from src.feature_engineering import add_time_features, categorize_tariff
from src.model_kmeans import perform_kmeans_clustering, evaluate_clustering, plot_clusters
from src.model_random_forest import train_random_forest
from src.model_neural_network import train_neural_network

def main():
    # -------------------------
    # 1. Data Preprocessing
    # -------------------------
    raw_data_path = "data/raw/smart_meter_data.csv"  # Replace with your raw data file path
    processed_data_path = "data/processed/smart_meter_data_processed.csv"
    
    print("Loading raw data...")
    data = load_data(raw_data_path)
    if data is None:
        return

    print("Handling missing values...")
    data = handle_missing_values(data)
    
    print("Removing outliers...")
    data = remove_outliers(data)
    
    print("Normalizing data...")
    data = normalize_data(data)
    
    data.to_csv(processed_data_path, index=False)
    print(f"Processed data saved to {processed_data_path}")
    
    # -------------------------
    # 2. Feature Engineering
    # -------------------------
    print("Adding time features...")
    data = add_time_features(data, timestamp_col="timestamp")
    
    print("Categorizing tariff...")
    data = categorize_tariff(data, tariff_col="tariff")
    
    features_data_path = "data/processed/smart_meter_data_features.csv"
    data.to_csv(features_data_path, index=False)
    print(f"Feature engineered data saved to {features_data_path}")
    
    # -------------------------
    # 3. Clustering (Unsupervised)
    # -------------------------
    print("Performing K-Means clustering...")
    # Use specific features if available; adjust based on your dataset
    if 'energy_consumption' in data.columns and 'hour' in data.columns:
        clustering_features = ['energy_consumption', 'hour']
    else:
        clustering_features = data.columns.tolist()[:2]
    
    data, kmeans_model = perform_kmeans_clustering(data, clustering_features, n_clusters=3)
    evaluate_clustering(data, clustering_features)
    plot_clusters(data, clustering_features, kmeans_model)
    
    joblib.dump(kmeans_model, "models/kmeans_model.pkl")
    print("K-Means model saved to models/kmeans_model.pkl")
    
    # -------------------------
    # 4. Supervised Learning (If anomaly labels exist)
    # -------------------------
    target = "anomaly"
    if target in data.columns:
        # Exclude columns that aren't features
        feature_columns = [col for col in data.columns if col not in ["timestamp", target, "cluster"]]
        
        # Train Random Forest Classifier
        print("Training Random Forest model...")
        rf_model = train_random_forest(data, feature_columns, target)
        joblib.dump(rf_model, "models/rf_model.pkl")
        print("Random Forest model saved to models/rf_model.pkl")
        
        # Train Neural Network Classifier
        print("Training Neural Network model...")
        nn_model, history, X_test, y_test, scaler = train_neural_network(data, feature_columns, target, epochs=20)
        nn_model.save("models/nn_model.h5")
        joblib.dump(scaler, "models/nn_scaler.pkl")
        print("Neural Network model saved to models/nn_model.h5")
        print("Scaler saved to models/nn_scaler.pkl")
    else:
        print("Target column 'anomaly' not found. Skipping supervised model training.")

if __name__ == "__main__":
    main()
