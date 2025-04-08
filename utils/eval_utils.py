# utils/eval_utils.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import torch
import os
import config
from scipy import stats

def classify_predictions(preds, threshold=0.5):
    """
    Convert probability predictions to binary class predictions
    
    Args:
        preds (numpy.ndarray): Predicted probabilities
        threshold (float, optional): Classification threshold
        
    Returns:
        numpy.ndarray: Binary class predictions
    """
    return (preds >= threshold).astype(int)

def calculate_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Calculate classification metrics
    
    Args:
        y_true (numpy.ndarray): Ground truth labels
        y_pred_proba (numpy.ndarray): Predicted probabilities
        threshold (float, optional): Classification threshold
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert probabilities to binary predictions
    y_pred = classify_predictions(y_pred_proba, threshold)
    
    # Calculate metrics
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['specificity'] = recall_score(1-y_true, 1-y_pred, zero_division=0)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    metrics['roc_auc'] = auc(fpr, tpr)
    metrics['fpr'] = fpr
    metrics['tpr'] = tpr
    
    # Calculate precision-recall curve and AUC
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    metrics['pr_auc'] = auc(recall, precision)
    metrics['precision_curve'] = precision
    metrics['recall_curve'] = recall
    
    # Calculate confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    return metrics

def evaluate_model(model, dataloader, criterion, device):
    """
    Evaluate model on the given dataloader
    
    Args:
        model (nn.Module): Model to evaluate
        dataloader (DataLoader): Data loader
        criterion (nn.Module): Loss function
        device (str): Device to evaluate on
        
    Returns:
        tuple: (loss, predictions, targets)
    """
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            # Move data to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1))
            
            # Update statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Store predictions and targets
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
    
    # Calculate loss
    avg_loss = running_loss / len(dataloader.dataset)
    
    return avg_loss, np.array(all_preds), np.array(all_targets)

def plot_roc_curve(metrics, save_path=None):
    """
    Plot ROC curve
    
    Args:
        metrics (dict): Metrics dictionary containing 'fpr', 'tpr', and 'roc_auc'
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['fpr'], metrics['tpr'], label=f"AUC = {metrics['roc_auc']:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path)
        print(f"ROC curve plot saved to {save_path}")
    
    plt.show()

def plot_precision_recall_curve(metrics, save_path=None):
    """
    Plot precision-recall curve
    
    Args:
        metrics (dict): Metrics dictionary containing 'precision_curve', 'recall_curve', and 'pr_auc'
        save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(8, 6))
    plt.plot(metrics['recall_curve'], metrics['precision_curve'], label=f"AUC = {metrics['pr_auc']:.4f}")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Precision-recall curve plot saved to {save_path}")
    
    plt.show()

def plot_confusion_matrix(metrics, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        metrics (dict): Metrics dictionary containing 'confusion_matrix'
        save_path (str, optional): Path to save the plot
    """
    cm = metrics['confusion_matrix']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved to {save_path}")
    
    plt.show()

def compare_models(reference_metrics, our_metrics, task):
    """
    Compare our model with the reference model
    
    Args:
        reference_metrics (dict): Metrics for the reference model
        our_metrics (dict): Metrics for our model
        task (str): Task name for the title
        
    Returns:
        tuple: (comparison_table, is_better)
    """
    # Define metrics to compare
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('Sensitivity', 'sensitivity'),
        ('Specificity', 'specificity'),
        ('ROC-AUC', 'roc_auc')
    ]
    
    # Create comparison table
    comparison_table = {
        'Metric': [m[0] for m in metrics_to_compare],
        'Reference': [reference_metrics.get(m[1], 'N/A') for m in metrics_to_compare],
        'Our Model': [our_metrics.get(m[1], 'N/A') for m in metrics_to_compare]
    }
    
    # Check if our model is better overall
    our_values = [our_metrics.get(m[1], 0) for m in metrics_to_compare]
    ref_values = [reference_metrics.get(m[1], 0) for m in metrics_to_compare]
    is_better = sum(our > ref for our, ref in zip(our_values, ref_values)) > len(metrics_to_compare) / 2
    
    return comparison_table, is_better

def paired_t_test(our_metrics_list, reference_value, metric='accuracy'):
    """
    Perform paired t-test to compare our model with the reference model
    
    Args:
        our_metrics_list (list): List of metrics dictionaries from k-fold cross-validation
        reference_value (float): Reference value to compare against
        metric (str, optional): Metric to compare
        
    Returns:
        tuple: (t_statistic, p_value, is_significant)
    """
    # Extract metric values from our model's k-fold results
    our_values = [metrics[metric] for metrics in our_metrics_list]
    
    # Create an array of reference values with the same length
    ref_values = np.full_like(our_values, reference_value)
    
    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(our_values, ref_values)
    
    # Check if the difference is significant (p < 0.05)
    is_significant = p_value < 0.05
    
    return t_stat, p_value, is_significant