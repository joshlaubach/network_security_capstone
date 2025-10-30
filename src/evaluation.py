"""
evaluation.py
-------------
Comprehensive evaluation metrics for both supervised and unsupervised models.

Includes:
  - Classification metrics (accuracy, precision, recall, F1, ROC-AUC)
  - Confusion matrix visualization
  - ROC curves
  - Precision-Recall curves
  - Classification reports
  - Anomaly detection metrics (for unsupervised)
  - Model comparison utilities
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    roc_curve, auc, roc_auc_score,
    precision_recall_curve, average_precision_score
)
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# Supervised Model Evaluation
# ================================================================

def evaluate_classification(y_true, y_pred, y_pred_proba=None, labels=None, 
                           model_name='Model', print_report=True):
    """
    Comprehensive evaluation for classification models.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities (optional, for ROC/PR curves)
        labels: List of class labels (optional)
        model_name: Name of the model for display
        print_report: Whether to print classification report
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Determine if binary or multi-class
    n_classes = len(np.unique(y_true))
    is_binary = n_classes == 2
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    if is_binary:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC (if probabilities provided)
        if y_pred_proba is not None:
            if len(y_pred_proba.shape) > 1:
                y_pred_proba = y_pred_proba[:, 1]  # Probability of positive class
            roc_auc = roc_auc_score(y_true, y_pred_proba)
            avg_precision = average_precision_score(y_true, y_pred_proba)
        else:
            roc_auc = None
            avg_precision = None
    else:
        # Multi-class metrics
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Multi-class ROC-AUC
        if y_pred_proba is not None:
            roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
            avg_precision = None  # Not typically used for multi-class
        else:
            roc_auc = None
            avg_precision = None
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'avg_precision': avg_precision,
        'confusion_matrix': cm,
        'n_classes': n_classes,
        'is_binary': is_binary
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"Evaluation Results: {model_name}")
    print("="*60)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    if roc_auc is not None:
        print(f"ROC-AUC:   {roc_auc:.4f}")
    if avg_precision is not None:
        print(f"Avg Precision: {avg_precision:.4f}")
    print("="*60)
    
    if print_report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=labels, zero_division=0))
    
    return results


def plot_confusion_matrix(y_true, y_pred, labels=None, normalize=False, 
                         title='Confusion Matrix', cmap='Blues', figsize=(8, 6)):
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        labels: List of class labels for display
        normalize: Whether to normalize the matrix
        title: Plot title
        cmap: Colormap for heatmap
        figsize: Figure size
    """
    # Convert to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title += ' (Normalized)'
    else:
        fmt = 'd'
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, square=True,
                xticklabels=labels, yticklabels=labels,
                cbar_kws={'label': 'Count' if not normalize else 'Proportion'})
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, model_name='Model', figsize=(8, 6)):
    """
    Plot ROC curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model for display
        figsize: Figure size
    """
    # Convert to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    # Extract probability of positive class
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds, 'auc': roc_auc}


def plot_precision_recall_curve(y_true, y_pred_proba, model_name='Model', figsize=(8, 6)):
    """
    Plot Precision-Recall curve for binary classification.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        model_name: Name of the model for display
        figsize: Figure size
    """
    # Convert to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    # Extract probability of positive class
    if len(y_pred_proba.shape) > 1:
        y_pred_proba = y_pred_proba[:, 1]
    
    # Compute PR curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = average_precision_score(y_true, y_pred_proba)
    
    # Plot
    plt.figure(figsize=figsize)
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f'Precision-Recall Curve - {model_name}', fontsize=14, fontweight='bold')
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return {'precision': precision, 'recall': recall, 'thresholds': thresholds, 
            'avg_precision': avg_precision}


def compare_roc_curves(models_dict, X_test, y_test, figsize=(10, 7)):
    """
    Plot ROC curves for multiple models on the same plot.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    
    for name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)
        
        # Extract probability of positive class
        if len(y_pred_proba.shape) > 1:
            y_pred_proba = y_pred_proba[:, 1]
        
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ================================================================
# Unsupervised Model Evaluation (Anomaly Detection)
# ================================================================

def evaluate_anomaly_detection(y_true, y_pred, y_scores=None, model_name='Model'):
    """
    Evaluate anomaly detection performance.
    
    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_pred: Predicted labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores (optional, for threshold analysis)
        model_name: Name of the model for display
        
    Returns:
        Dictionary of evaluation metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix components
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Anomaly-specific metrics
    false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
    true_negative_rate = tn / (tn + fp) if (tn + fp) > 0 else 0  # Specificity
    
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,  # Also known as True Positive Rate
        'f1_score': f1,
        'false_positive_rate': false_positive_rate,
        'true_negative_rate': true_negative_rate,
        'true_positives': tp,
        'false_positives': fp,
        'true_negatives': tn,
        'false_negatives': fn
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"Anomaly Detection Results: {model_name}")
    print("="*60)
    print(f"Accuracy:     {accuracy:.4f}")
    print(f"Precision:    {precision:.4f}")
    print(f"Recall (TPR): {recall:.4f}")
    print(f"F1-Score:     {f1:.4f}")
    print(f"FPR:          {false_positive_rate:.4f}")
    print(f"TNR:          {true_negative_rate:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp:6d}  |  FP: {fp:6d}")
    print(f"  FN: {fn:6d}  |  TN: {tn:6d}")
    print("="*60)
    
    return results


def plot_anomaly_score_distribution(y_true, y_scores, model_name='Model', 
                                    threshold=None, figsize=(10, 6), ax=None):
    """
    Plot distribution of anomaly scores for normal vs anomalous samples.
    
    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores
        model_name: Name of the model for display
        threshold: Decision threshold (optional)
        figsize: Figure size
        ax: Matplotlib axes object (optional). If provided, plot on this axes instead of creating new figure
    """
    # Convert to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_scores, pd.Series):
        y_scores = y_scores.values
    
    # Separate scores by true label
    normal_scores = y_scores[y_true == 0]
    anomaly_scores = y_scores[y_true == 1]
    
    # Create figure if ax is not provided
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
    
    # Plot histograms
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', color='red', density=True)
    
    # Plot threshold if provided
    if threshold is not None:
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.3f}')
    
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Anomaly Score Distribution - {model_name}', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Only call tight_layout and show if we created our own figure
    if ax is plt.gca():
        plt.tight_layout()
        plt.show()


def find_optimal_threshold(y_true, y_scores, metric='f1', plot=True):
    """
    Find optimal threshold for anomaly detection based on a metric.
    
    Args:
        y_true: True labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        plot: Whether to plot metric vs threshold
        
    Returns:
        Dictionary with optimal threshold and metric value
    """
    # Convert to numpy if needed
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_scores, pd.Series):
        y_scores = y_scores.values
    
    # Try different thresholds
    thresholds = np.percentile(y_scores, np.linspace(0, 100, 100))
    metric_values = []
    
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(int)
        
        if metric == 'f1':
            val = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            val = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            val = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            val = accuracy_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        metric_values.append(val)
    
    # Find optimal threshold
    optimal_idx = np.argmax(metric_values)
    optimal_threshold = thresholds[optimal_idx]
    optimal_value = metric_values[optimal_idx]
    
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, metric_values, linewidth=2, color='blue')
        plt.axvline(x=optimal_threshold, color='red', linestyle='--', linewidth=2,
                   label=f'Optimal threshold = {optimal_threshold:.3f}')
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel(metric.upper(), fontsize=12)
        plt.title(f'Threshold Optimization - {metric.upper()}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    print(f"\nOptimal Threshold: {optimal_threshold:.4f}")
    print(f"Optimal {metric.upper()}: {optimal_value:.4f}")
    
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_value': optimal_value,
        'metric': metric,
        'all_thresholds': thresholds,
        'all_values': metric_values
    }


# ================================================================
# Model Comparison
# ================================================================

def compare_models_table(results_list):
    """
    Create comparison table from multiple evaluation results.
    
    Args:
        results_list: List of evaluation result dictionaries
        
    Returns:
        DataFrame with model comparison
    """
    comparison_data = []
    
    for result in results_list:
        row = {
            'Model': result['model_name'],
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score']
        }
        
        if 'roc_auc' in result and result['roc_auc'] is not None:
            row['ROC-AUC'] = result['roc_auc']
        
        if 'false_positive_rate' in result:
            row['FPR'] = result['false_positive_rate']
        
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    print("\n" + "="*80)
    print("Model Comparison Table")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80 + "\n")
    
    return df


def plot_metrics_comparison(comparison_df, metrics=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                           figsize=(12, 6)):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        comparison_df: DataFrame from compare_models_table
        metrics: List of metrics to plot
        figsize: Figure size
    """
    # Filter to only include metrics that exist in the DataFrame
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    if not available_metrics:
        print("[WARN] No metrics available to plot")
        return
    
    x = np.arange(len(comparison_df))
    width = 0.8 / len(available_metrics)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple']
    
    for i, metric in enumerate(available_metrics):
        offset = width * i - (width * len(available_metrics) / 2) + width / 2
        ax.bar(x + offset, comparison_df[metric], width, 
               label=metric, color=colors[i % len(colors)])
    
    ax.set_xlabel('Model', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    ax.legend(fontsize=10)
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def create_evaluation_report(models_dict, X_test, y_test, report_type='supervised',
                             save_path=None):
    """
    Create comprehensive evaluation report for multiple models.
    
    Args:
        models_dict: Dictionary of {model_name: model}
        X_test: Test features
        y_test: Test labels
        report_type: 'supervised' or 'unsupervised'
        save_path: Optional path to save report
        
    Returns:
        Dictionary with all evaluation results
    """
    all_results = []
    
    print("\n" + "="*80)
    print(f"Creating Evaluation Report - {report_type.upper()}")
    print("="*80)
    
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        
        if report_type == 'supervised':
            y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
            result = evaluate_classification(y_test, y_pred, y_pred_proba, model_name=name)
        else:  # unsupervised/anomaly detection
            y_scores = model.decision_function(X_test) if hasattr(model, 'decision_function') else None
            result = evaluate_anomaly_detection(y_test, y_pred, y_scores, model_name=name)
        
        all_results.append(result)
    
    # Create comparison table
    comparison_df = compare_models_table(all_results)
    
    # Plot comparison
    plot_metrics_comparison(comparison_df)
    
    report = {
        'individual_results': all_results,
        'comparison_table': comparison_df,
        'report_type': report_type
    }
    
    # Save if path provided
    if save_path:
        comparison_df.to_csv(save_path, index=False)
        print(f"\n[INFO] Report saved to {save_path}")
    
    return report


# ================================================================
# Utility Functions
# ================================================================

def print_summary_stats(y_true, dataset_name='Dataset'):
    """
    Print summary statistics of the dataset.
    
    Args:
        y_true: True labels
        dataset_name: Name of the dataset
    """
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    
    unique, counts = np.unique(y_true, return_counts=True)
    
    print("\n" + "="*60)
    print(f"Dataset Summary: {dataset_name}")
    print("="*60)
    print(f"Total samples: {len(y_true)}")
    print(f"Number of classes: {len(unique)}")
    print("\nClass distribution:")
    for label, count in zip(unique, counts):
        percentage = 100 * count / len(y_true)
        print(f"  Class {label}: {count:6d} ({percentage:5.2f}%)")
    print("="*60 + "\n")
