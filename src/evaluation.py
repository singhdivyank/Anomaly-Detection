import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    silhouette_score
)

def evaluate_classification_model(y_true, y_pred, y_prob=None):
    """
    Evaluate a classification model using various metrics.
    
    Parameters:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        y_prob (array-like, optional): Predicted probabilities for the positive class.
                                       If provided, the ROC curve and AUC will be computed.
    
    Returns:
        dict: A dictionary containing accuracy, confusion matrix, classification report,
              and ROC AUC (if probabilities are provided).
    """
    results = {}
    results['accuracy'] = accuracy_score(y_true, y_pred)
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    results['classification_report'] = classification_report(y_true, y_pred)
    
    print("Accuracy: {:.3f}".format(results['accuracy']))
    print("Confusion Matrix:")
    print(results['confusion_matrix'])
    print("Classification Report:")
    print(results['classification_report'])
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        results['roc_auc'] = roc_auc
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.show()
        
    return results

def evaluate_clustering_model(data, features, cluster_col='cluster'):
    """
    Evaluate clustering performance using the silhouette score.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the clustering results.
        features (list): List of features used for clustering.
        cluster_col (str): Column name with cluster labels.
        
    Returns:
        float: The silhouette score of the clustering.
    """
    if cluster_col not in data.columns:
        print(f"Column '{cluster_col}' not found in data.")
        return None
    
    score = silhouette_score(data[features], data[cluster_col])
    print(f"Silhouette Score: {score:.3f}")
    return score

if __name__ == "__main__":
    # Example usage for classification evaluation
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 0]
    y_prob = [0.2, 0.8, 0.3, 0.4, 0.9, 0.1]
    evaluate_classification_model(y_true, y_pred, y_prob)
