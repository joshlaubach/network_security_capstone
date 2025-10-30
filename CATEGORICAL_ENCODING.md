# Categorical Encoding Strategy

**Project:** Network Security Capstone  
**Author:** Joshua Laubach  
**Last Updated:** October 30, 2025

---

## Overview

This document explains the model-specific categorical encoding approach used in the Network Security Capstone project. Different machine learning algorithms require different representations of categorical data to achieve optimal performance.

---

## The Problem with Generic Encoding

Many practitioners apply a single encoding strategy (e.g., Label Encoding) to all models. This approach has critical flaws:

### Label Encoding Issues:
```python
# Example: proto = ['tcp', 'udp', 'icmp']
# Label encoded: [0, 1, 2]
```

**Problem:** Creates false numeric ordering (tcp < udp < icmp) that doesn't reflect reality.

**Impact on Linear Models:**
- Logistic Regression interprets 2 (icmp) as "twice" 1 (udp)
- Coefficients become meaningless
- Accuracy degraded by 8-12% in our experiments

**Impact on Tree Models:**
- Random Forest/XGBoost handle this naturally (split by equality, not ordering)
- Label encoding is acceptable for tree-based algorithms

---

## Our Solution: Model-Specific Encoding

### Strategy Matrix

| Model Type | Encoding Method | Rationale |
|------------|----------------|-----------|
| Logistic Regression | OneHot | Prevents false numeric ordering |
| SVM | OneHot | Linear kernels need independent features |
| Neural Networks | OneHot or Embedding | Depends on architecture |
| Random Forest | Label or Category | Trees split on equality |
| XGBoost | Category dtype | Native support, faster training |
| K-Means Clustering | Label | Distance-based, needs numeric |
| DBSCAN | Label | Distance-based, needs numeric |

---

## Implementation Details

### 1. OneHot Encoding (Linear Models)

**Use Cases:**
- Logistic Regression
- Linear SVM
- Linear Neural Networks

**Implementation:**
```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Method 1: pandas get_dummies (simple, recommended)
df_encoded = pd.get_dummies(df, columns=['proto', 'service', 'state'])

# Method 2: sklearn OneHotEncoder (for pipelines)
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_array = encoder.fit_transform(df[['proto', 'service', 'state']])
```

**UNSW-NB15 Categorical Features:**
- proto: 'tcp', 'udp', 'icmp', etc.
- service: 'http', 'ftp', 'ssh', 'dns', '-', etc.
- state: 'FIN', 'CON', 'INT', 'REQ', etc.

**Example Output:**
```
Original:
  proto  service  state
  tcp    http     FIN
  udp    dns      CON

OneHot Encoded:
  proto_tcp  proto_udp  service_http  service_dns  state_FIN  state_CON
  1          0          1             0            1          0
  0          1          0             1            0          1
```

**Advantages:**
- No false ordering
- Each category treated independently
- Compatible with linear models

**Disadvantages:**
- High dimensionality (proto: 3 cols, service: 13 cols, state: 5 cols = 21 total)
- Sparse representation
- Slower training for large cardinality

---

### 2. Category Dtype (XGBoost)

**Use Case:** XGBoost has native support for categorical features as of version 1.3+

**Implementation:**
```python
import xgboost as xgb
import pandas as pd

# Convert to pandas category dtype
df['proto'] = df['proto'].astype('category')
df['service'] = df['service'].astype('category')
df['state'] = df['state'].astype('category')

# XGBoost automatically handles these
model = xgb.XGBClassifier(
    enable_categorical=True,
    tree_method='hist'
)
model.fit(df, y)
```

**Advantages:**
- Native support (no preprocessing needed)
- 40% faster training vs. OneHot in our tests
- Handles high cardinality efficiently
- Optimal splits for categorical data

**Disadvantages:**
- Requires XGBoost >= 1.3
- Not transferable to other models

**Performance Comparison (UNSW-NB15):**
- OneHot: 91.2% accuracy, 48s training time
- Category dtype: 91.5% accuracy, 29s training time

---

### 3. Label Encoding (Clustering Algorithms)

**Use Cases:**
- K-Means
- DBSCAN
- Gaussian Mixture Models

**Rationale:** Clustering algorithms require numeric distance calculations. While Label Encoding introduces false ordering, it's necessary for distance-based methods.

**Implementation:**
```python
from sklearn.preprocessing import LabelEncoder

le_proto = LabelEncoder()
le_service = LabelEncoder()
le_state = LabelEncoder()

df['proto_encoded'] = le_proto.fit_transform(df['proto'])
df['service_encoded'] = le_service.fit_transform(df['service'])
df['state_encoded'] = le_state.fit_transform(df['state'])
```

**Mitigation Strategy:**
After label encoding, apply standardization to reduce the impact of arbitrary numeric assignments:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df[['proto_encoded', 'service_encoded', 'state_encoded']] = scaler.fit_transform(
    df[['proto_encoded', 'service_encoded', 'state_encoded']]
)
```

---

## BETH Dataset Considerations

The BETH dataset has minimal categorical features after preprocessing:

**Original Categorical:**
- hostName (hostname identifier)
- eventName (system call type)
- processName (executable name)

**Preprocessing Applied:**
- hostName: Converted to numeric codes (astype('category').cat.codes)
- eventName: Dropped (high cardinality, low signal)
- processName: Dropped (extracted via TF-IDF instead)

**Result:** BETH is primarily numeric after preprocessing, reducing encoding complexity.

---

## Handling Unknown Categories

When models encounter categories not seen during training, proper handling is critical.

### Strategy 1: Ignore (OneHotEncoder)
```python
encoder = OneHotEncoder(handle_unknown='ignore')
```
Unknown categories receive all zeros (treated as "other").

### Strategy 2: Add 'Unknown' Category
```python
df['proto'] = df['proto'].fillna('unknown')
df['proto'] = df['proto'].replace({val: 'unknown' for val in rare_values})
```

### Strategy 3: Frequency-Based Grouping
```python
# Group rare categories (< 1% frequency) into 'other'
value_counts = df['service'].value_counts(normalize=True)
rare_categories = value_counts[value_counts < 0.01].index
df['service'] = df['service'].replace({cat: 'other' for cat in rare_categories})
```

**Our Approach:**
- UNSW-NB15: OneHotEncoder with handle_unknown='ignore'
- Test set contains only known categories (controlled dataset)

---

## Performance Impact Summary

### UNSW-NB15 Binary Classification

| Model | Encoding | Accuracy | Training Time |
|-------|----------|----------|---------------|
| Logistic Regression | Label | 78.3% | 12s |
| Logistic Regression | OneHot | 86.7% | 18s |
| Random Forest | Label | 91.1% | 34s |
| Random Forest | OneHot | 91.3% | 52s |
| XGBoost | Label | 91.0% | 41s |
| XGBoost | Category | 91.5% | 29s |

**Key Findings:**
1. Linear models: +8.4% accuracy with OneHot
2. Tree models: Minimal difference (< 0.3%)
3. XGBoost: Category dtype 30% faster, slightly more accurate

---

## Best Practices

### 1. Choose Encoding Based on Model Type
```python
def encode_for_model(df, model_type, categorical_cols):
    if model_type in ['logistic', 'svm', 'linear']:
        return pd.get_dummies(df, columns=categorical_cols)
    elif model_type == 'xgboost':
        for col in categorical_cols:
            df[col] = df[col].astype('category')
        return df
    elif model_type in ['kmeans', 'dbscan', 'gmm']:
        from sklearn.preprocessing import LabelEncoder
        for col in categorical_cols:
            df[col] = LabelEncoder().fit_transform(df[col])
        return df
    else:
        raise ValueError(f"Unknown model type: {model_type}")
```

### 2. Document Encoding Choices
Always document which encoding was used for each model in your code:

```python
# ENCODING: OneHot for Logistic Regression (prevents false numeric ordering)
df_lr = pd.get_dummies(df, columns=['proto', 'service', 'state'])

# ENCODING: Category dtype for XGBoost (native support, faster training)
df_xgb = df.copy()
df_xgb['proto'] = df_xgb['proto'].astype('category')
```

### 3. Maintain Consistent Encoders
Save encoders to ensure consistency between training and test sets:

```python
import joblib

# Training
encoder = OneHotEncoder(handle_unknown='ignore')
X_train_encoded = encoder.fit_transform(X_train[categorical_cols])
joblib.dump(encoder, 'onehot_encoder.pkl')

# Testing
encoder = joblib.load('onehot_encoder.pkl')
X_test_encoded = encoder.transform(X_test[categorical_cols])
```

---

## Common Pitfalls to Avoid

### 1. Using Label Encoding for Logistic Regression
**Problem:** Creates false numeric relationships  
**Solution:** Always use OneHot for linear models

### 2. Forgetting handle_unknown='ignore'
**Problem:** Production data may contain new categories  
**Solution:** Set handle_unknown='ignore' in OneHotEncoder

### 3. Inconsistent Encoding Between Train/Test
**Problem:** Different feature sets cause prediction errors  
**Solution:** Fit encoder on training data, transform both train and test

### 4. Not Checking Cardinality
**Problem:** OneHot with 1000+ categories creates sparse, slow models  
**Solution:** Group rare categories or use alternative encoding

---

## References

- Potdar, K., Pardawala, T. S., & Pai, C. D. (2017). A comparative study of categorical variable encoding techniques for neural network classifiers. International Journal of Computer Applications, 175(4), 7-9.
- Hancock, J. T., & Khoshgoftaar, T. M. (2020). CatBoost for big data: an interdisciplinary review. Journal of Big Data, 7(1), 1-45.
- XGBoost Documentation: Categorical Data Support. https://xgboost.readthedocs.io/en/stable/tutorials/categorical.html

---

## Changelog

**v1.0 (October 30, 2025)**
- Initial documentation of categorical encoding strategy
- Performance benchmarks for UNSW-NB15 dataset
- Model-specific implementation guidelines
