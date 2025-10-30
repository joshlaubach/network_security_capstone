"""
Generate UNSW-NB15 supervised model comparison figure
4-panel layout matching BETH dataset visualization style
All metrics shown as percentages with 2 decimal places
NO ANNOTATIONS - Clean graphs for PowerPoint editing

NOTE: These results are from the TEST SET (82,332 samples)
Source: notebooks/01_unsw_supervised.ipynb - final evaluation
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for presentations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk", font_scale=1.3)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Create figures directory
figures_dir = Path('figures/presentation')
figures_dir.mkdir(parents=True, exist_ok=True)

# Load actual results
results_df = pd.read_csv('results/unsw_supervised_comparison.csv')

print("UNSW-NB15 Dataset: Supervised Model Comparison")
print("="*70)
print("\nLoaded data:")
print(results_df)
print()

# Model colors
MODEL_COLORS = {
    'Logistic Regression': '#2ECC71',  # Green
    'Random Forest': '#3498DB',        # Blue
    'XGBoost': '#E67E22'               # Orange
}

# Prepare data for visualization
models = ['Logistic Regression', 'Random Forest', 'XGBoost']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Extract values from CSV
model_data = {
    'Logistic Regression': results_df[results_df['model'] == 'logistic_regression'].iloc[0],
    'Random Forest': results_df[results_df['model'] == 'random_forest'].iloc[0],
    'XGBoost': results_df[results_df['model'] == 'xgboost'].iloc[0]
}

# Map CSV columns to metrics
metric_mapping = {
    'Accuracy': 'accuracy',
    'Precision': 'precision',
    'Recall': 'recall',
    'F1-Score': 'f1_score'
}

print("Creating 4-panel comparison figure...")
print()

# Create figure with subplots for each metric (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Get values for each model (convert to percentage)
    csv_col = metric_mapping[metric]
    values = [model_data[m][csv_col] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.6
    
    # Create bars with model-specific colors
    bars = ax.bar(x, values, width, 
                  color=[MODEL_COLORS[m] for m in models], 
                  alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars with 2 decimal places
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{val:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_ylabel(f'{metric} (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(['Logistic\nRegression', 'Random\nForest', 'XGBoost'], 
                       fontsize=13, fontweight='bold')
    
    # Set y-axis to percentage scale with appropriate range
    if metric == 'Precision':
        ax.set_ylim([80, 102])  # Precision is very high
    elif metric == 'Recall':
        ax.set_ylim([80, 92])   # Recall is lower
    elif metric == 'Accuracy':
        ax.set_ylim([80, 95])   # Accuracy mid-range
    else:  # F1-Score
        ax.set_ylim([80, 95])
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('UNSW-NB15 Dataset: Supervised Model Performance Comparison\nBinary & Multi-Class Attack Classification (Test Set)', 
             fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
output_file = figures_dir / 'slide3_supervised_4panel_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
print(f"  Size: {output_file.stat().st_size / 1024:.0f} KB")
plt.close()

print("\n" + "="*70)
print("✓ UNSW-NB15 supervised comparison figure generated!")
print(f"✓ Location: {output_file.absolute()}")
print("="*70)
print("\nFigure shows:")
print("  • 4-panel layout (Accuracy, Precision, Recall, F1-Score)")
print("  • All metrics as percentages with 2 decimals")
print("  • Model-specific colors (Green=LR, Blue=RF, Orange=XGB)")
print("  • Clean graphs (no annotations)")
print("  • Test set results (82,332 samples)")
print("  • Matches BETH figure style exactly")
print("\nReady for Slide 3 in your presentation!")
