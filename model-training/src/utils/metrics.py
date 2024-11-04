import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def calculate_precision(y_true, y_pred):
    """
    Calculates the precision score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Precision score.
    """
    return precision_score(y_true, y_pred, average='weighted')

def calculate_recall(y_true, y_pred):
    """
    Calculates the recall score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Recall score.
    """
    return recall_score(y_true, y_pred, average='weighted')

def calculate_f1(y_true, y_pred):
    """
    Calculates the F1 score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - F1 score.
    """
    return f1_score(y_true, y_pred, average='weighted')

def calculate_accuracy(y_true, y_pred):
    """
    Calculates the accuracy score.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.

    Returns:
    - Accuracy score.
    """
    return accuracy_score(y_true, y_pred)

def calculate_auc(y_true, y_scores):
    """
    Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC) score.

    Parameters:
    - y_true: True labels.
    - y_scores: Predicted scores or probabilities.

    Returns:
    - ROC AUC score.
    """
    return roc_auc_score(y_true, y_scores)

def calculate_all_metrics(y_true, y_pred, y_scores=None):
    """
    Calculates all major metrics: precision, recall, F1, accuracy, and optionally ROC AUC.

    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - y_scores: Predicted scores or probabilities (required for AUC).

    Returns:
    - A dictionary containing all calculated metrics.
    """
    metrics = {
        'precision': calculate_precision(y_true, y_pred),
        'recall': calculate_recall(y_true, y_pred),
        'f1_score': calculate_f1(y_true, y_pred),
        'accuracy': calculate_accuracy(y_true, y_pred)
    }
    
    if y_scores is not None:
        metrics['roc_auc'] = calculate_auc(y_true, y_scores)

    return metrics