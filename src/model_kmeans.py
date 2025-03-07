import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def perform_kmeans_clustering(data, features, n_clusters=3, random_state=42):
    """
    Perform K-means clustering on the given data using the specified features.
    
    Args:
        data (pd.DataFrame): DataFrame containing the data.
        features (list): List of column names to be used for clustering.
        n_clusters (int): Number of clusters to form.
        random_state (int): Seed for random number generator.
    
    Returns:
        tuple: DataFrame with an additional 'cluster' column and the trained KMeans model.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    data['cluster'] = kmeans.fit_predict(data[features])
    return data, kmeans

def evaluate_clustering(data, features):
    """
    Evaluate the clustering performance using the silhouette score.
    
    Args:
        data (pd.DataFrame): DataFrame with a 'cluster' column.
        features (list): List of features used for clustering.
    
    Returns:
        float: The silhouette score.
    """
    score = silhouette_score(data[features], data['cluster'])
    print(f"Silhouette Score: {score:.3f}")
    return score

def plot_clusters(data, features, kmeans):
    """
    Plot clusters based on the first two features provided.
    
    Args:
        data (pd.DataFrame): DataFrame with clustering results.
        features (list): List of feature names used for clustering (at least two required).
        kmeans (KMeans): Trained KMeans model.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(data[features[0]], data[features[1]], c=data['cluster'], cmap='viridis', alpha=0.6)
    
    # Plot centroids
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Centroids')
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.title("K-Means Clustering")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Example usage:
    # Load the feature-engineered data
    data_path = "../data/processed/smart_meter_data_features.csv"
    model_output_path = "../models/kmeans_model.pkl"
    
    data = pd.read_csv(data_path)
    
    # Specify the features to use for clustering.
    # Adjust these columns based on your dataset; here we assume 'energy_consumption' and 'hour' exist.
    if 'energy_consumption' in data.columns and 'hour' in data.columns:
        features = ['energy_consumption', 'hour']
    else:
        features = data.columns.tolist()[:2]  # fallback to first two columns
    
    # Perform clustering
    data, kmeans = perform_kmeans_clustering(data, features, n_clusters=3)
    
    # Evaluate clustering performance
    evaluate_clustering(data, features)
    
    # Plot the clusters if at least two features are available
    if len(features) >= 2:
        plot_clusters(data, features, kmeans)
    
    # Save the trained model for later use
    joblib.dump(kmeans, model_output_path)
    print(f"K-Means model saved to {model_output_path}.")
