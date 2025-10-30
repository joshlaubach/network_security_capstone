"""
Generate improved BETH dataset comparison figure
Optimized Models (Baseline) vs TF-IDF Enhanced Models
All metrics shown as percentages with 2 decimal places
NO ANNOTATIONS - Clean graphs for PowerPoint editing

NOTE: These results are from the TEST SET (188,967 samples)
Source: notebooks/02_beth_unsupervised.ipynb - "EVALUATE TUNED MODELS ON TEST SET"
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
results_df = pd.read_csv('results/beth_baseline_vs_enhanced_comparison.csv')

print("BETH Dataset: Optimized Models vs TF-IDF Enhanced Models")
print("="*70)
print("\nLoaded data:")
print(results_df)
print()

# Model colors
MODEL_COLORS = {
    'K-Means': '#F1C40F',  # Yellow
    'DBSCAN': '#9B59B6',   # Purple
    'GMM': '#E74C3C'       # Red
}

# Prepare data for visualization
models = ['K-Means', 'DBSCAN', 'GMM']
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']

# Extract baseline (Numeric Only) values
baseline_data = {
    'K-Means': results_df[results_df['Model'] == 'K-Means (Baseline)'].iloc[0],
    'DBSCAN': results_df[results_df['Model'] == 'DBSCAN (Baseline)'].iloc[0],
    'GMM': results_df[results_df['Model'] == 'GMM (Baseline)'].iloc[0]
}

# Extract enhanced (Numeric + TF-IDF) values
enhanced_data = {
    'K-Means': results_df[results_df['Model'] == 'K-Means (Enhanced)'].iloc[0],
    'DBSCAN': results_df[results_df['Model'] == 'DBSCAN (Enhanced)'].iloc[0],
    'GMM': results_df[results_df['Model'] == 'GMM (Enhanced)'].iloc[0]
}

print("Creating comprehensive comparison figure...")
print()

# Create figure with subplots for each metric
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, metric in enumerate(metrics):
    ax = axes[idx]
    
    # Get values for each model
    baseline_values = [baseline_data[m][metric] * 100 for m in models]  # Convert to percentage
    enhanced_values = [enhanced_data[m][metric] * 100 for m in models]  # Convert to percentage
    
    x = np.arange(len(models))
    width = 0.35
    
    # Create bars
    bars1 = ax.bar(x - width/2, baseline_values, width, 
                   label='Optimized (Numeric Only)', 
                   color=[MODEL_COLORS[m] for m in models], 
                   alpha=0.6, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, enhanced_values, width, 
                   label='TF-IDF Enhanced (Numeric + Text)', 
                   color=[MODEL_COLORS[m] for m in models], 
                   alpha=1.0, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars with 2 decimal places
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Formatting
    ax.set_ylabel(f'{metric} (%)', fontsize=14, fontweight='bold')
    ax.set_title(f'{metric} Comparison', fontsize=15, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13, fontweight='bold')
    
    # Legend in top right with rounded box
    legend = ax.legend(loc='upper right', fontsize=10, framealpha=0.95, 
                       fancybox=True, shadow=True, frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.5)
    
    # Set y-axis to percentage scale
    if metric == 'Recall':
        ax.set_ylim([92, 101])  # Recall is very high
    elif metric == 'Precision':
        ax.set_ylim([92, 101])  # Precision is very high
    else:
        ax.set_ylim([92, 98])
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.suptitle('BETH Dataset: Optimized Models vs TF-IDF Enhanced Models\nUnsupervised Anomaly Detection Performance (Test Set)', 
             fontsize=18, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.98])
output_file = figures_dir / 'slide4_beth_optimized_vs_tfidf.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# Create a single comprehensive side-by-side comparison
print("\nCreating single-panel comprehensive comparison...")

fig, ax = plt.subplots(figsize=(14, 8))

# Prepare data for grouped bar chart
n_models = len(models)
n_metrics = len(metrics)
x = np.arange(n_models)
width = 0.18  # Width of each bar

# Plot each metric as a separate bar group
metric_positions = [-1.5, -0.5, 0.5, 1.5]

for i, metric in enumerate(metrics):
    baseline_values = [baseline_data[m][metric] * 100 for m in models]
    enhanced_values = [enhanced_data[m][metric] * 100 for m in models]
    
    offset = metric_positions[i] * width
    
    # Baseline bars (lighter)
    ax.bar(x + offset - width/2, baseline_values, width*0.9, 
           alpha=0.5, edgecolor='black', linewidth=1,
           color=[MODEL_COLORS[m] for m in models])
    
    # Enhanced bars (darker with pattern)
    bars = ax.bar(x + offset + width/2, enhanced_values, width*0.9,
                  alpha=1.0, edgecolor='black', linewidth=1,
                  color=[MODEL_COLORS[m] for m in models],
                  label=metric if i < 4 else "")

# Custom legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='gray', alpha=0.5, edgecolor='black', label='Optimized (Numeric Only)'),
    Patch(facecolor='gray', alpha=1.0, edgecolor='black', label='TF-IDF Enhanced (Numeric + Text)'),
    Patch(facecolor='white', edgecolor='white', label=''),  # Spacer
    Patch(facecolor=MODEL_COLORS['K-Means'], label='K-Means'),
    Patch(facecolor=MODEL_COLORS['DBSCAN'], label='DBSCAN'),
    Patch(facecolor=MODEL_COLORS['GMM'], label='GMM'),
]

ax.legend(handles=legend_elements, loc='lower right', fontsize=11, 
          framealpha=0.95, ncol=2)

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_title('BETH Dataset: Optimized vs TF-IDF Enhanced Performance\nAll Metrics Comparison (Test Set)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=13, fontweight='bold')
ax.set_ylim([92, 101])
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add metric labels at bottom
metric_labels_x = []
for i, metric in enumerate(metrics):
    offset = metric_positions[i] * width
    metric_labels_x.append(x[1] + offset)  # Center on middle model

for i, (metric, pos) in enumerate(zip(metrics, metric_labels_x)):
    ax.text(pos, 91.5, metric, ha='center', fontsize=10, 
           fontweight='bold', style='italic',
           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.5))

plt.tight_layout()
output_file2 = figures_dir / 'slide4_beth_comprehensive_comparison.png'
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file2}")
plt.close()

# Create simple side-by-side for DBSCAN only (best model)
print("\nCreating DBSCAN-focused comparison (best performer)...")

fig, ax = plt.subplots(figsize=(12, 7))

metrics_display = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'FPR', 'TNR']
baseline_values = [
    baseline_data['DBSCAN']['Accuracy'] * 100,
    baseline_data['DBSCAN']['Precision'] * 100,
    baseline_data['DBSCAN']['Recall'] * 100,
    baseline_data['DBSCAN']['F1-Score'] * 100,
    baseline_data['DBSCAN']['FPR'] * 100,
    baseline_data['DBSCAN']['TNR'] * 100,
]

enhanced_values = [
    enhanced_data['DBSCAN']['Accuracy'] * 100,
    enhanced_data['DBSCAN']['Precision'] * 100,
    enhanced_data['DBSCAN']['Recall'] * 100,
    enhanced_data['DBSCAN']['F1-Score'] * 100,
    enhanced_data['DBSCAN']['FPR'] * 100,
    enhanced_data['DBSCAN']['TNR'] * 100,
]

x = np.arange(len(metrics_display))
width = 0.35

bars1 = ax.bar(x - width/2, baseline_values, width, 
               label='DBSCAN Optimized\n(Numeric Only)', 
               color='#9B59B6', alpha=0.6, edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x + width/2, enhanced_values, width, 
               label='DBSCAN TF-IDF Enhanced\n(Numeric + Text)', 
               color='#9B59B6', alpha=1.0, edgecolor='black', linewidth=1.5)

# Add value labels with 2 decimal places
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
               f'{height:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('BETH Dataset: DBSCAN Optimized vs TF-IDF Enhanced\nBest Performing Unsupervised Model (Test Set)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics_display, fontsize=12, fontweight='bold')

# Legend in top right with rounded box
legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
                   fancybox=True, shadow=True, frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.5)

ax.set_ylim([0, 110])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
output_file3 = figures_dir / 'slide4_beth_dbscan_comparison.png'
plt.savefig(output_file3, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file3}")
plt.close()

print("\n" + "="*70)
print("✓ All BETH comparison figures generated successfully!")
print(f"✓ Location: {figures_dir.absolute()}")
print("="*70)
print("\nFigures created:")
print("  1. slide4_beth_optimized_vs_tfidf.png (4-panel metric comparison)")
print("  2. slide4_beth_comprehensive_comparison.png (All models, all metrics)")
print("  3. slide4_beth_dbscan_comparison.png (DBSCAN-focused)")
print("\nAll values shown as percentages with 2 decimal places!")
print("NO ANNOTATIONS - Clean graphs ready for PowerPoint editing")
print("\nNOTE: These are TEST SET results (188,967 samples)")
print("RECOMMENDED FOR PRESENTATION:")
print("  → slide4_beth_dbscan_comparison.png (clean, focused, easy to annotate)")
