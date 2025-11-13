# Network Security Capstone Project

### Detecting and Classifying Cyber Threats with Machine Learning

**Author:** Joshua Laubach  
**Program:** Boston University – Online Master of Data Science (OMDS)  
**Course:** DX 799S O2, Data Science Capstone (Module C)  
**Instructor:** Professor Joshua Von Korff  
**Date:** October 27, 2025  

---

## Overview

This project fulfills the requirements for the **Module C Integrated Capstone**, bringing together technical, analytical, and communication skills from the OMDS program.  
The goal: build a reproducible, end-to-end data science pipeline that detects and classifies cybersecurity threats using real network and system-level data.

### Deliverables
1. **Technical Report (18–25 pages)** – detailed methods, modeling, and evaluation for a data-science audience.  
2. **Non-technical Report (8–12 pages)** – executive summary and actionable insights for stakeholders.  

Together they comprise 100 points (50% of course grade).

---

## Problem Definition

Cyber attacks evolve faster than traditional rule-based systems. This project applies machine learning to improve threat detection by analyzing both **network traffic** and **system call logs**.

**Objectives**
- Reduce false positives in intrusion detection.
- Identify novel or unseen attack types.
- Produce interpretable, reproducible results.

---

## Datasets

| Dataset | Type | Records | Task |
|----------|------|----------|------|
| **UNSW-NB15** | Network traffic flows | 257 K | Supervised classification |
| **BETH** | System-call logs | 1.14 M | Unsupervised anomaly detection |

---

## Methods

### Supervised Models (UNSW-NB15)
- Logistic Regression  
- Random Forest  
- XGBoost  
Each tuned with cross-validation and evaluated using Accuracy, Precision, Recall, F1, and ROC-AUC.

### Unsupervised Models (BETH)
- K-Means  
- DBSCAN  
- Gaussian Mixture Models  
Evaluated using Silhouette Score, Detection Rate, and False Positive Rate.

### Feature Engineering
- 65 derived network features (sum, diff, ratio).  
- TF-IDF vectorization of system-call arguments.  
- Six feature-selection strategies compared.  
- Log transforms to correct extreme skewness.

---

## Project Structure

```
network_security_capstone/
│
├── src/                    # Core modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models_supervised.py
│   └── models_unsupervised.py
│
├── notebooks/              # Analysis & visualization
│   ├── 01_data_overview.ipynb
│   ├── 02_beth_unsupervised.ipynb
│   ├── 03_unsw_supervised.ipynb
│   └── 04_results_comparison.ipynb
│
├── data/                   # Generated after download
├── results/                # Model outputs
└── figures/                # Visualizations
```

---

## Setup

```bash
git clone https://github.com/yourusername/network_security_capstone.git
cd network_security_capstone
python -m venv venv
source venv/bin/activate     # macOS/Linux
pip install -r requirements.txt
```

### Download Datasets

Configure your Kaggle API credentials, then run:

```bash
python src/data_download.py
```

This downloads and extracts both UNSW-NB15 and BETH datasets to the `data/` directory.

---

## Running the Analysis

Execute notebooks in order:

1. **01_data_overview.ipynb** - Exploratory data analysis and initial visualizations
2. **02_beth_unsupervised.ipynb** - Anomaly detection with K-Means, DBSCAN, GMM
3. **03_unsw_supervised.ipynb** - Classification with Logistic Regression, Random Forest, XGBoost
4. **04_results_comparison.ipynb** - Cross-dataset comparison and final insights
5. **05_presentation_visuals.ipynb** - Publication-ready figures and final presentation materials

---

## Key Results

**Supervised Classification (UNSW-NB15):**
- Best model: Random Forest
- Accuracy: 90.61%
- Precision: 98.69%
- Recall: 87.36%
- F1-Score: 92.68%

**Model Comparison:**
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 87.42% | 98.16% | 83.07% | 89.99% |
| Random Forest | 90.61% | 98.69% | 87.36% | 92.68% |
| XGBoost | 90.05% | 98.94% | 86.31% | 92.19% |

**Unsupervised Anomaly Detection (BETH):**
- Best clustering method: DBSCAN (Enhanced with TF-IDF)
- Accuracy: 96.01%
- Precision: 95.90%
- Recall: 99.87%
- F1-Score: 97.84%
- False positive rate: 41.80%
- True negative rate: 58.20%

**BETH Model Comparison:**
| Model | Feature Set | Accuracy | Precision | Recall | F1-Score | FPR |
|-------|-------------|----------|-----------|--------|----------|-----|
| DBSCAN (Baseline) | Numeric Only | 94.32% | 99.67% | 94.05% | 96.78% | 3.06% |
| DBSCAN (Enhanced) | Numeric + TF-IDF | 96.01% | 95.90% | 99.87% | 97.84% | 41.80% |
| K-Means (Enhanced) | Numeric + TF-IDF | 93.28% | 98.10% | 94.42% | 96.22% | 17.91% |

**Feature Engineering Impact:**
- Original features: 49 (UNSW), 17 base (BETH)
- Engineered features: 118 (UNSW), 500+ with TF-IDF (BETH)
- After selection: 30 (UNSW), 50 (BETH)
- Dimensionality reduction: 75.21% (UNSW)
- Performance retention: >99%

**Two-Stage Pipeline (UNSW-NB15):**
- Stage 1: Binary attack detection (Normal vs Attack)
- Stage 2: Multi-class attack type classification
- End-to-end accuracy: 74.74%
- Realistic security operations workflow

---

## Highlights

### Technical Innovations

1. **Hybrid Detection Framework**
   - Supervised learning for known attack patterns
   - Unsupervised learning for novel anomaly discovery
   - Complementary approaches for comprehensive coverage

2. **Advanced Feature Engineering**
   - TF-IDF extraction from system call arguments (500 features)
   - 65 derived network traffic features (sum, diff, ratio, zero flags)
   - Integer ratio detection for gridded pattern identification
   - Log transformations with explicit naming (`log_` prefix)

3. **Systematic Feature Selection**
   - 6 methods compared (Variance, Correlation, RFE, Model-Based, L1, Mutual Info)
   - Dimensionality reduction: 105 → 30 features (71% reduction)
   - Performance vs. efficiency trade-off analysis

4. **Model-Specific Categorical Encoding**
   - OneHot for linear models and clustering
   - Category dtype for XGBoost native support
   - Preserves interpretability and prevents false numeric ordering

### Reproducibility Features

- Fixed random seeds (42) across all experiments
- Modular code architecture (preprocessing, models, evaluation separate)
- Comprehensive documentation (5 notebooks + technical docs)
- Preserved train/test splits for consistent evaluation

---

## Documentation

- **FEATURE_IMPROVEMENTS.md** - Recent feature engineering enhancements
- **CATEGORICAL_ENCODING.md** - Model-specific encoding strategy details
- **Notebooks** - Step-by-step analysis with visualizations

---

## Citation

```
Laubach, J. (2025). Detecting and Classifying Cyber Threats with Machine Learning.
Boston University Online Master of Data Science, Module C Capstone Project.
Instructor: Professor Joshua Von Korff.
```

---

## Acknowledgments

- **Professor Joshua Von Korff** - Course instructor and capstone advisor
- **Boston University OMDS Program** - Academic foundation
- **UNSW Canberra Cyber Range Lab** - UNSW-NB15 dataset (Moustafa & Slay, 2015)
- **BETH Dataset Contributors** - Honeypot system call logs via Kaggle
- **Open Source Community** - scikit-learn, XGBoost, pandas, NumPy, matplotlib

---

**For academic purposes only. Do not use for unauthorized network monitoring or offensive security research.**
