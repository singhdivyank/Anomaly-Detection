import pandas as pd
import joblib

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.data_preprocessing import handle_missing_values, remove_outliers, normalize_data
from src.model_kmeans import perform_kmeans_clustering, evaluate_clustering, plot_clusters
from src.consts import *

def load_data(file_path):
    try:
        data = pd.read_csv(file_path, skipinitialspace=True)
        columns = data.columns.str.strip()
        # Convert energy column
        if 'KWH/hh (per half hour)' in columns:
            data['KWH/hh (per half hour)'] = pd.to_numeric(data['KWH/hh (per half hour)'], errors='coerce')
        # Convert tariff column if exists
        if 'tariff' in columns:
            data['tariff'] = pd.to_numeric(data['tariff'], errors='coerce')
        print(f"Data loaded successfully from {file_path}.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess(data):
    print("Handling missing values...")
    data = handle_missing_values(data)
    print("Removing outliers...")
    data = remove_outliers(data)
    print("Normalizing data...")
    data = normalize_data(data)
    return data

def evaluate_model(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

def perform_clustering(data):
    print("Performing K-Means clustering...")
    clustering_features = ['energy_consumption', 'hour'] if 'energy_consumption' in data.columns and 'hour' in data.columns else data.columns.tolist()[:2]
    
    data, kmeans_model = perform_kmeans_clustering(data, clustering_features, n_clusters=3)
    evaluate_clustering(data, clustering_features)
    plot_clusters(data, clustering_features, kmeans_model)
    
    joblib.dump(kmeans_model, KMEANS_PATH)
    print(f"K-Means model saved to {KMEANS_PATH}")