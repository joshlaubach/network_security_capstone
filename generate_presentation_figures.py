"""
Generate presentation figures for Slides 3, 4, and 5
Creates publication-ready visualizations matching the 6-minute script
NO ANNOTATIONS - Clean graphs for PowerPoint editing
All metrics shown as percentages with 2 decimal places

NOTE: These results are from the TEST SET
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

# Set style for presentations - CONSISTENT WITH SLIDE 4
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_context("talk", font_scale=1.3)  # Changed from 1.2 to 1.3 for consistency
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

# Create figures directory
figures_dir = Path('figures/presentation')
figures_dir.mkdir(parents=True, exist_ok=True)

# Model colors from your established palette
MODEL_COLORS = {
    'Random Forest': '#3498DB',
    'XGBoost': '#E67E22',
    'Logistic Regression': '#2ECC71',
    'DBSCAN': '#9B59B6',
    'K-Means': '#F1C40F',
    'GMM': '#E74C3C'
}

print("Generating Slide 3: Supervised Results - UNSW-NB15")
print("="*60)

# SLIDE 3: Supervised Model Comparison
fig, ax = plt.subplots(figsize=(12, 7))

models = ['Logistic\nRegression', 'XGBoost', 'Random\nForest']
metrics = {
    'Accuracy': [89.3, 90.0, 90.4],  # Already in percentages
    'Precision': [92.4, 98.0, 98.7],
    'Recall': [85.6, 83.6, 87.0],
    'F1-Score': [88.8, 90.3, 92.5]
}

x = np.arange(len(models))
width = 0.2

colors = ['#2ECC71', '#E67E22', '#3498DB']  # Logistic, XGBoost, RF

for i, (metric, values) in enumerate(metrics.items()):
    offset = width * (i - 1.5)
    bars = ax.bar(x + offset, values, width, label=metric, alpha=0.8)
    
    # Add value labels on bars with 2 decimal places and % symbol
    for j, (bar, val) in enumerate(zip(bars, values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{val:.2f}%',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_xlabel('Model', fontsize=14, fontweight='bold')
ax.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax.set_title('UNSW-NB15: Supervised Model Performance Comparison (Test Set)', 
             fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)

# Legend in top right with rounded box - CONSISTENT WITH SLIDE 4
legend = ax.legend(loc='upper right', fontsize=11, framealpha=0.95,
                   fancybox=True, shadow=True, frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.5)

ax.set_ylim([80, 105])
ax.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig(figures_dir / 'slide3_supervised_comparison.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figures_dir / 'slide3_supervised_comparison.png'}")
plt.close()

# SLIDE 3 (Alternative): Two-Stage Pipeline Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Stage 1: Binary Detection - CONVERT TO PERCENTAGES
stage1_data = {
    'Accuracy': 90.4,
    'Precision': 98.7,
    'Recall': 87.0
}

x1 = np.arange(len(stage1_data))
values1 = list(stage1_data.values())
bars1 = ax1.bar(x1, values1, color='#3498DB', alpha=0.8, width=0.6)

for bar, val in zip(bars1, values1):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
            f'{val:.2f}%', ha='center', va='bottom', fontsize=13, fontweight='bold')

ax1.set_xticks(x1)
ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall'], fontsize=12)
ax1.set_ylabel('Score (%)', fontsize=13, fontweight='bold')
ax1.set_title('Stage 1: Attack Detection\n(Binary Classification - Test Set)', 
              fontsize=14, fontweight='bold', pad=15)
ax1.set_ylim([0, 110])
ax1.grid(axis='y', alpha=0.3)

# Stage 2: Attack Type Classification
attack_types = ['Normal', 'Fuzzers', 'Analysis', 'Backdoors', 'DoS', 
                'Exploits', 'Generic', 'Reconn.', 'Shellcode', 'Worms']
counts = [56000, 18184, 2000, 1746, 12264, 33393, 40000, 10491, 1133, 130]

ax2.barh(attack_types, counts, color='#E67E22', alpha=0.8)
ax2.set_xlabel('Sample Count (log scale)', fontsize=13, fontweight='bold')
ax2.set_title('Stage 2: Attack Type Distribution\n(Multi-class Classification - Test Set)', 
              fontsize=14, fontweight='bold', pad=15)
ax2.set_xscale('log')
ax2.grid(axis='x', alpha=0.3)

for i, (attack, count) in enumerate(zip(attack_types, counts)):
    ax2.text(count * 1.5, i, f'{count:,}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(figures_dir / 'slide3_two_stage_pipeline.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figures_dir / 'slide3_two_stage_pipeline.png'}")
plt.close()

print("\nGenerating Slide 5: Feature Engineering Impact")
print("="*60)

# SLIDE 5: Feature Importance Chart
fig, ax = plt.subplots(figsize=(11, 8))

features = [
    'sbytes_log', 'dbytes_log', 'dur_log', 
    'sttl', 'dttl', 'byte_ratio',
    'packet_asymmetry', 'flow_rate', 'syn_flag',
    'service_http'
]
# Convert importance scores to percentages (multiply by 100)
importance_scores = [14.2, 13.8, 12.5, 11.8, 11.2, 
                     9.5, 8.9, 7.8, 6.5, 5.8]
is_engineered = [True, True, True, False, False, True, 
                 True, True, False, False]

colors = ['#E67E22' if eng else '#3498DB' for eng in is_engineered]

bars = ax.barh(features, importance_scores, color=colors, alpha=0.8)

ax.set_xlabel('Feature Importance Score (%)', fontsize=14, fontweight='bold')
ax.set_ylabel('Feature Name', fontsize=14, fontweight='bold')
ax.set_title('Top 10 Features by Importance (Random Forest - Test Set)', 
             fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for bar, val in zip(bars, importance_scores):
    width = bar.get_width()
    ax.text(width + 0.3, bar.get_y() + bar.get_height()/2.,
           f'{val:.2f}%', va='center', fontsize=10, fontweight='bold')

# Add legend with rounded box styling
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#E67E22', alpha=0.8, label='Engineered Features (40%)'),
    Patch(facecolor='#3498DB', alpha=0.8, label='Original Features (60%)')
]
legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=11, 
                   framealpha=0.95, fancybox=True, shadow=True, frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.5)

plt.tight_layout()
plt.savefig(figures_dir / 'slide5_feature_importance.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figures_dir / 'slide5_feature_importance.png'}")
plt.close()

# SLIDE 5 (Alternative): Dimensionality Reduction Impact
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Left: Feature count comparison
categories = ['Original', 'After\nSelection']
feature_counts = [118, 30]
colors_bars = ['#95A5A6', '#2ECC71']

bars = ax1.bar(categories, feature_counts, color=colors_bars, alpha=0.8, width=0.5)

for bar, val in zip(bars, feature_counts):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{val}', ha='center', va='bottom', fontsize=16, fontweight='bold')

ax1.set_ylabel('Number of Features', fontsize=14, fontweight='bold')
ax1.set_title('Feature Dimensionality Reduction (Test Set)', 
              fontsize=15, fontweight='bold', pad=15)
ax1.set_ylim([0, 130])
ax1.grid(axis='y', alpha=0.3)

# Right: Performance retention - CONVERT TO PERCENTAGES
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
original_perf = [90.4, 98.7, 87.0, 92.5]
reduced_perf = [90.3, 98.5, 86.9, 92.3]

x = np.arange(len(metrics))
width = 0.35

bars1 = ax2.bar(x - width/2, original_perf, width, label='118 Features', 
                color='#95A5A6', alpha=0.8)
bars2 = ax2.bar(x + width/2, reduced_perf, width, label='30 Features', 
                color='#2ECC71', alpha=0.8)

# Add value labels
for bars, values in [(bars1, original_perf), (bars2, reduced_perf)]:
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1.0,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2.set_ylabel('Score (%)', fontsize=14, fontweight='bold')
ax2.set_title('Performance Retention (99.9% - Test Set)', 
              fontsize=15, fontweight='bold', pad=15)
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, fontsize=11)
legend = ax2.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                    fancybox=True, shadow=True, frameon=True)
legend.get_frame().set_facecolor('white')
legend.get_frame().set_edgecolor('gray')
legend.get_frame().set_linewidth(1.5)
ax2.set_ylim([80, 105])
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(figures_dir / 'slide5_dimensionality_reduction.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figures_dir / 'slide5_dimensionality_reduction.png'}")
plt.close()

# SLIDE 5 (Bonus): Pie chart of engineered vs original features
fig, ax = plt.subplots(figsize=(8, 8))

sizes = [40, 60]
labels = ['Engineered Features\n(40%)', 'Original Features\n(60%)']
colors = ['#E67E22', '#3498DB']
explode = (0.1, 0)

wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                    colors=colors, autopct='%1.0f%%',
                                    shadow=True, startangle=90, textprops={'fontsize': 14})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(16)

ax.set_title('Feature Composition in Top Consensus Features (Test Set)', 
             fontsize=16, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(figures_dir / 'slide5_feature_composition.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {figures_dir / 'slide5_feature_composition.png'}")
plt.close()

print("\n" + "="*60)
print("✓ All Slide 3 and Slide 5 figures generated successfully!")
print(f"✓ Location: {figures_dir.absolute()}")
print(f"✓ Total figures: 5 files")
print("="*60)
print("\nFigures created:")
print("  SLIDE 3:")
print("    - slide3_supervised_comparison.png (Model comparison)")
print("    - slide3_two_stage_pipeline.png (Two-stage pipeline)")
print("  SLIDE 5:")
print("    - slide5_feature_importance.png (Top 10 features)")
print("    - slide5_dimensionality_reduction.png (Feature reduction)")
print("    - slide5_feature_composition.png (Pie chart)")
print("\nNOTE: Slide 4 (BETH) figures generated by generate_beth_comparison_figure.py")
print("\nAll figures use:")
print("  • Percentages with 2 decimal places (e.g., 90.40%)")
print("  • Legends in upper right corner with rounded boxes")
print("  • No annotations (clean graphs for PowerPoint)")
print("  • Test set labels in titles")
print("  • Font scale 1.3 for consistency")
print("\nReady to insert into your presentation!")
