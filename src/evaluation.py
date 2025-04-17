import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, 
    confusion_matrix, 
    classification_report, 
    roc_curve, 
    auc, 
    silhouette_score
)
from src.utils import plot_roc

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
    results = {
        'accuracy': 0.0,
        'confusion_matrix': 0.0,
        'classification_report': 0.0,
        'roc_auc': 0.0
    }

    accuracy, cm, class_report = accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred), classification_report(y_true, y_pred)
    print("Accuracy: {:.3f}".format(accuracy))
    print("Confusion Matrix: \n", cm)
    print("Classification Report: \n", class_report)
    
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plot_roc(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
        results.update({
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'roc_auc': roc_auc
        })
        
    return results

def evaluate_clustering_model(data, features):
    """
    Evaluate clustering performance using the silhouette score.
    
    Parameters:
        data (pd.DataFrame): DataFrame containing the clustering results.
        features (list): List of features used for clustering.
        
    Returns:
        float: The silhouette score of the clustering.
    """
    if 'cluster' not in data.columns:
        print("Column 'cluster' not found in data.")
        return None
    
    score = silhouette_score(data[features], data['cluster'])
    print(f"Silhouette Score: {score:.3f}")
    return score

if __name__ == "__main__":
    evaluate_classification_model(y_true=[0, 1, 0, 1, 1, 0], y_pred=[0, 1, 0, 0, 1, 0], y_prob=[0.2, 0.8, 0.3, 0.4, 0.9, 0.1])
