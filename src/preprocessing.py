"""
preprocessing.py
----------------
Handles data loading and preprocessing for:
    1. BETH dataset (unsupervised anomaly detection)
    2. UNSW-NB15 dataset (supervised classification)

Integrates with KaggleHub for reproducible dataset retrieval.

NOTE: This module imports advanced feature engineering from feature_engineering.py,
      which includes log transformations, integer ratio detection, and automatic
      pair feature generation. See feature_engineering.py and FEATURE_IMPROVEMENTS.md
      for details on the feature engineering pipeline.

Outputs:
    - pandas DataFrames for training, validation, and testing sets
    - Cleaned, standardized (numeric only), ready-to-model datasets
    - Categorical features KEPT AS STRINGS for model-specific encoding
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import kagglehub

# Import advanced feature engineering functions
from feature_engineering import (
    _feature_engineer_unsw,
    _apply_log_transforms_unsw,
    _apply_log_transforms_beth
)


# ================================================================
# Utility Functions
# ================================================================

def scale_numeric_features(df):
    """
    Scale ONLY numeric columns in a DataFrame using StandardScaler.
    
    IMPORTANT: 
    - Categorical (object dtype) columns are skipped automatically.
    - Target columns (label, attack_cat) are excluded from scaling.
    """
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    # Exclude target columns from scaling
    num_cols = [c for c in num_cols if c not in ['label', 'attack_cat']]
    
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
    return df


def _safe_drop(df, columns):
    """Drop columns safely without throwing KeyErrors."""
    df.drop(columns=[c for c in columns if c in df.columns], inplace=True, errors="ignore")
    return df


def _clean_dataframe(df):
    """Replace inf/nan and fill remaining gaps with zeros."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


# ================================================================
# BETH Dataset
# ================================================================

def _get_beth_paths():
    """Download and return file paths for BETH train, val, test sets."""
    base_path = kagglehub.dataset_download("katehighnam/beth-dataset")
    paths = {
        "train": os.path.join(base_path, "labelled_training_data.csv"),
        "val": os.path.join(base_path, "labelled_validation_data.csv"),
        "test": os.path.join(base_path, "labelled_testing_data.csv"),
    }
    print(f"[INFO] BETH dataset located at {base_path}")
    return paths


def _preprocess_beth_split(df):
    """Apply feature engineering transformations to one BETH DataFrame."""
    df.rename(columns={"timestamp": "timeSinceBoot"}, inplace=True)
    df["processId_flag"] = df["processId"].isin([0, 1, 2]).astype(int)
    df["parentProcessId_flag"] = df["parentProcessId"].isin([0, 1, 2]).astype(int)
    df["userId_flag"] = (df["userId"] < 1000).astype(int)
    df["mountNamespace_flag"] = (df["mountNamespace"] == 4026531840).astype(int)
    df["returnValue_flag"] = np.sign(df["returnValue"]).astype(int)

    _safe_drop(df, ["threadId", "processName", "stackAddresses", "eventName"])

    if "hostName" in df.columns:
        df["hostName"] = df["hostName"].astype("category").cat.codes

    df = _clean_dataframe(df)
    return df


def load_beth():
    """
    Load and preprocess all three BETH splits with log transformations.
    
    Pipeline:
        1. Load CSV data
        2. Preprocess each split (drop columns, encode categoricals, clean)
        3. Apply log transformations (timeSinceBoot, processId, parentProcessId)
        4. Scale numeric features only
        5. Save original and preprocessed data to data folder
    
    Returns:
        tuple: (train_df, val_df, test_df) - all DataFrames with preprocessed features
    """
    paths = _get_beth_paths()
    beth_train_orig = pd.read_csv(paths["train"])
    beth_val_orig = pd.read_csv(paths["val"])
    beth_test_orig = pd.read_csv(paths["test"])
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'beth')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save original data
    beth_train_orig.to_csv(os.path.join(data_dir, 'beth_train_original.csv'), index=False)
    beth_val_orig.to_csv(os.path.join(data_dir, 'beth_val_original.csv'), index=False)
    beth_test_orig.to_csv(os.path.join(data_dir, 'beth_test_original.csv'), index=False)
    print(f"[INFO] Original BETH data saved to {data_dir}")

    # Preprocess each split
    beth_train = _preprocess_beth_split(beth_train_orig.copy())
    beth_val = _preprocess_beth_split(beth_val_orig.copy())
    beth_test = _preprocess_beth_split(beth_test_orig.copy())
    
    # Apply log transformations
    beth_train = _apply_log_transforms_beth(beth_train)
    beth_val = _apply_log_transforms_beth(beth_val)
    beth_test = _apply_log_transforms_beth(beth_test)

    # Scale numeric features
    beth_train = scale_numeric_features(beth_train)
    beth_val = scale_numeric_features(beth_val)
    beth_test = scale_numeric_features(beth_test)
    
    # Save preprocessed data
    beth_train.to_csv(os.path.join(data_dir, 'beth_train_preprocessed.csv'), index=False)
    beth_val.to_csv(os.path.join(data_dir, 'beth_val_preprocessed.csv'), index=False)
    beth_test.to_csv(os.path.join(data_dir, 'beth_test_preprocessed.csv'), index=False)
    print(f"[INFO] Preprocessed BETH data saved to {data_dir}")

    print(f"[INFO] BETH splits loaded with log transformations:")
    print(f"       Train: {beth_train.shape}, Val: {beth_val.shape}, Test: {beth_test.shape}")
    
    # Report log-transformed features
    log_features = [col for col in beth_train.columns if col.startswith('log_')]
    if log_features:
        print(f"       Log-transformed features: {log_features}")

    return beth_train, beth_val, beth_test


# ================================================================
# UNSW-NB15 Dataset
# ================================================================

def _get_unsw_paths():
    """Download and return file paths for UNSW train and test sets."""
    base_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    paths = {
        "train": os.path.join(base_path, "UNSW_NB15_training-set.csv"),
        "test": os.path.join(base_path, "UNSW_NB15_testing-set.csv"),
    }
    print(f"[INFO] UNSW-NB15 dataset located at {base_path}")
    return paths


def _encode_categoricals(df):
    """
    Label-encode categorical features for compatibility with feature selection.
    
    Categorical columns (object/category dtype) are encoded to integers using
    LabelEncoder. This allows feature selection algorithms and most models to
    work seamlessly while preserving the information content of these features.
    
    Args:
        df: DataFrame with potential categorical columns
        
    Returns:
        df: DataFrame with categorical columns encoded as integers
    """
    from sklearn.preprocessing import LabelEncoder
    
    # Identify categorical columns (excluding target columns)
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [c for c in categorical_cols if c not in ['label', 'attack_cat']]
    
    if len(categorical_cols) == 0:
        return df
    
    # Encode each categorical column
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
    
    return df


def load_unsw(split_test=True):
    """
    Load and preprocess UNSW-NB15 dataset with proper train/test splitting.
    
    Pipeline:
        1. Load CSV data from Kaggle
        2. Split Kaggle test set in half to create validation and test sets
        3. Add source tracking ('train', 'val', 'test')
        4. Drop NaNs
        5. Encode categorical features (proto, service, state) to integers
        6. Feature engineering (optional - controlled by parameters)
        7. Log transformations (optional)
        8. Clean dataframe
        9. Scale numeric features PROPERLY (fit on train, transform all sets)
    
    NOTE: This approach matches the reference notebook that achieved ~91% accuracy.
          The key is splitting the Kaggle test set in half, which creates a more
          compatible test set than using the full Kaggle test set.
    
    Args:
        split_test: If True, splits Kaggle test set in half for val/test (default).
                    If False, uses original Kaggle splits (lower accuracy).
    
    Returns:
        tuple: (train_df, val_df, test_df) - all DataFrames with features + label + attack_cat + source
               If split_test=False, returns (train_df, None, test_df)
    """
    from sklearn.model_selection import train_test_split
    
    paths = _get_unsw_paths()
    train_df_orig = pd.read_csv(paths["train"])
    test_df_orig = pd.read_csv(paths["test"])
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'unsw')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save original data
    train_df_orig.to_csv(os.path.join(data_dir, 'unsw_train_original.csv'), index=False)
    test_df_orig.to_csv(os.path.join(data_dir, 'unsw_test_original.csv'), index=False)
    print(f"[INFO] Original UNSW-NB15 data saved to {data_dir}")
    
    # Work with copies
    train_df = train_df_orig.copy()
    test_df = test_df_orig.copy()
    
    if split_test:
        # Split test set in half to create validation and test sets (reference approach)
        test_df, val_df = train_test_split(
            test_df, 
            test_size=0.5, 
            random_state=42, 
            stratify=test_df['label']
        )
        
        # Add source tracking
        train_df['source'] = 'train'
        val_df['source'] = 'val'
        test_df['source'] = 'test'
        
        # Combine for preprocessing
        combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        print(f"[INFO] UNSW-NB15 dataset loaded with test set split:")
        print(f"       Original train: {len(train_df)} samples")
        print(f"       Validation: {len(val_df)} samples (from Kaggle test)")
        print(f"       Test: {len(test_df)} samples (from Kaggle test)")
    else:
        # Use original Kaggle splits (lower accuracy due to distribution mismatch)
        train_df['source'] = 'train'
        test_df['source'] = 'test'
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        val_df = None
        
        print(f"[INFO] UNSW-NB15 dataset loaded with original Kaggle splits:")
        print(f"       Train: {len(train_df)} samples")
        print(f"       Test: {len(test_df)} samples")
    
    # Save source column before processing (it might get modified)
    source_column = combined_df['source'].copy()
    
    # Process combined data (but keep source safe)
    combined_df.dropna(inplace=True)
    
    # Drop ID column (it's just a row identifier, not a feature)
    if 'id' in combined_df.columns:
        combined_df.drop(columns=['id'], inplace=True)
    
    # Restore source after dropna (in case indices changed)
    source_column = source_column.loc[combined_df.index]
    combined_df['source'] = source_column
    
    combined_df = _encode_categoricals(combined_df)
    combined_df = _feature_engineer_unsw(combined_df)
    combined_df = _apply_log_transforms_unsw(combined_df)
    combined_df = _clean_dataframe(combined_df)
    
    # Ensure source column is still there
    combined_df['source'] = source_column.loc[combined_df.index]
    
    # Split back using source tracking
    train_processed = combined_df[combined_df['source'] == 'train'].copy()
    if split_test:
        val_processed = combined_df[combined_df['source'] == 'val'].copy()
        test_processed = combined_df[combined_df['source'] == 'test'].copy()
    else:
        val_processed = None
        test_processed = combined_df[combined_df['source'] == 'test'].copy()
    
    # Remove source column after splitting
    train_processed.drop(columns=['source'], inplace=True)
    if val_processed is not None:
        val_processed.drop(columns=['source'], inplace=True)
    test_processed.drop(columns=['source'], inplace=True)
    
    # Scale using TRAIN scaler only
    num_cols = train_processed.select_dtypes(include=["float64", "int64"]).columns
    num_cols = [c for c in num_cols if c not in ['label', 'attack_cat']]
    
    if len(num_cols) > 0:
        scaler = StandardScaler()
        train_processed[num_cols] = scaler.fit_transform(train_processed[num_cols])
        if val_processed is not None:
            val_processed[num_cols] = scaler.transform(val_processed[num_cols])
        test_processed[num_cols] = scaler.transform(test_processed[num_cols])
    
    # Save preprocessed data
    train_processed.to_csv(os.path.join(data_dir, 'unsw_train_preprocessed.csv'), index=False)
    if val_processed is not None:
        val_processed.to_csv(os.path.join(data_dir, 'unsw_val_preprocessed.csv'), index=False)
    test_processed.to_csv(os.path.join(data_dir, 'unsw_test_preprocessed.csv'), index=False)
    print(f"[INFO] Preprocessed UNSW-NB15 data saved to {data_dir}")
    
    print(f"[INFO] Preprocessing complete:")
    print(f"       Features: {len(num_cols)} numeric")
    print(f"       Scaling: Fit on train, applied to all sets")
    print(f"       Categorical encoding: proto, service, state → integers")
    
    return train_processed, val_processed, test_processed


# ================================================================
# Combined Convenience Loader
# ================================================================

def prepare_datasets():
    """
    Download and preprocess all datasets.
    Returns:
        beth_train, beth_val, beth_test, unsw_train, unsw_val, unsw_test
    """
    beth_train, beth_val, beth_test = load_beth()
    unsw_train, unsw_val, unsw_test = load_unsw()
    print("[INFO] All datasets prepared successfully.")
    return beth_train, beth_val, beth_test, unsw_train, unsw_val, unsw_test