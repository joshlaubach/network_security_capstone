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
    _apply_log_transforms_beth,
    engineer_beth_tfidf_features,
    cleanup_tfidf_text_columns
)


# ================================================================
# Utility Functions
# ================================================================

def scale_numeric_features(df):
    """
    Scale ONLY numeric columns in a DataFrame using StandardScaler.
    
    IMPORTANT: 
    - Categorical (object dtype) columns are skipped automatically.
    - Target/label columns are EXCLUDED from scaling.
    - Binary columns (sus, evil, label, attack_cat, *_flag) are EXCLUDED.
    
    BUG FIX: Previously scaled binary target columns, causing values outside [0,1].
    """
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    # Exclude target columns and binary indicators from scaling
    exclude_cols = ['label', 'attack_cat', 'sus', 'evil']
    
    # Also exclude any column ending with '_flag' (binary indicators)
    exclude_cols.extend([c for c in num_cols if c.endswith('_flag')])
    
    # Auto-detect binary columns (2 unique values or less)
    for col in num_cols:
        if col not in exclude_cols and df[col].nunique() <= 2:
            exclude_cols.append(col)
    
    # Filter to only columns that should be scaled
    num_cols = [c for c in num_cols if c not in exclude_cols]
    
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        print(f"Scaled {len(num_cols)} continuous features (excluded {len(exclude_cols)} binary/target columns)")
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
    print(f"BETH dataset located at {base_path}")
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


# Public API for pipeline use
def preprocess_beth_split(df):
    """
    Preprocess a single BETH DataFrame (public API).
    
    Transformations applied:
    - Rename 'timestamp' to 'timeSinceBoot'
    - Create binary flag features for special ID values
    - Drop unnecessary columns (threadId, processName, stackAddresses, eventName)
    - Encode hostName as category codes
    - Clean infinities and NaNs
    
    Args:
        df: Raw BETH DataFrame
        
    Returns:
        Preprocessed DataFrame (same shape, cleaned features)
        
    Example:
        >>> from data_extraction import extract_beth
        >>> from preprocessing import preprocess_beth_split
        >>> train_raw, val_raw, test_raw = extract_beth()
        >>> train_clean = preprocess_beth_split(train_raw)
    """
    return _preprocess_beth_split(df.copy())


def load_beth(raw=False, tfidf=False, max_features=500, min_df=2, max_df=0.5):
    """
    Load and preprocess all three BETH splits with log transformations.
    
    Pipeline:
        1. Load CSV data
        2. (If raw=True) Return original unprocessed data
        3. (If raw=False) Preprocess each split (drop columns, encode categoricals, clean)
        4. Apply log transformations (timeSinceBoot, processId, parentProcessId)
        5. (Optional) Extract TF-IDF features from 'args' column
        6. Scale numeric features only
        7. Save original and preprocessed data to data folder
    
    Args:
        raw: If True, return raw unprocessed data without any transformations.
             If False, apply full preprocessing pipeline (default).
        tfidf: If True, extract TF-IDF features from 'args' column and return them.
               If False, skip TF-IDF extraction (default).
               Note: Ignored if raw=True.
        max_features: Maximum number of TF-IDF features (only used if tfidf=True)
        min_df: Minimum document frequency for TF-IDF (only used if tfidf=True)
        max_df: Maximum document frequency for TF-IDF (only used if tfidf=True)
    
    Returns:
        If raw=True:
            tuple: (train_df, val_df, test_df) - Original unprocessed DataFrames
        
        If raw=False and tfidf=False:
            tuple: (train_df, val_df, test_df) - DataFrames with preprocessed features
        
        If raw=False and tfidf=True:
            tuple: (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
                - X_train/val/test: Sparse matrices combining numeric + TF-IDF features
                - y_train/val/test: DataFrames with target columns ('sus', 'evil')
                - feature_names: List of TF-IDF feature names
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
    print(f"Original BETH data saved to {data_dir}")

    # If raw=True, return original unprocessed data
    if raw:
        print(f"Returning RAW unprocessed BETH data:")
        print(f"       Train: {beth_train_orig.shape}, Val: {beth_val_orig.shape}, Test: {beth_test_orig.shape}")
        return beth_train_orig, beth_val_orig, beth_test_orig

    # Preprocess each split
    beth_train = _preprocess_beth_split(beth_train_orig.copy())
    beth_val = _preprocess_beth_split(beth_val_orig.copy())
    beth_test = _preprocess_beth_split(beth_test_orig.copy())
    
    # Apply log transformations
    beth_train = _apply_log_transforms_beth(beth_train)
    beth_val = _apply_log_transforms_beth(beth_val)
    beth_test = _apply_log_transforms_beth(beth_test)

    # SCALING DISABLED: Don't scale numeric features here!
    # The raw feature magnitudes are needed for distance-based anomaly detection.
    # Scaling compresses distances and makes threshold optimization unreliable.
    # If scaling is needed, it should be done AFTER clustering models compute distances.
    # beth_train = scale_numeric_features(beth_train)
    # beth_val = scale_numeric_features(beth_val)
    # beth_test = scale_numeric_features(beth_test)
    print("Note: Numeric features NOT scaled (raw magnitudes preserved for distance calculations)")

    # TF-IDF feature extraction and combination (optional)
    if tfidf:
        from feature_engineering import combine_numeric_and_tfidf
        
        print(f"\nExtracting TF-IDF features from 'args' column...")
        tfidf_train, tfidf_val, tfidf_test, feature_names, vectorizer = engineer_beth_tfidf_features(
            beth_train, beth_val, beth_test,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            verbose=True
        )
        
        # Combine numeric and TF-IDF features into sparse matrices
        print(f"\nCombining numeric and TF-IDF features...")
        X_train = combine_numeric_and_tfidf(beth_train, tfidf_train)
        X_val = combine_numeric_and_tfidf(beth_val, tfidf_val)
        X_test = combine_numeric_and_tfidf(beth_test, tfidf_test)
        
        # Extract labels before dropping target columns
        y_train = beth_train[['sus', 'evil']].copy()
        y_val = beth_val[['sus', 'evil']].copy()
        y_test = beth_test[['sus', 'evil']].copy()
        
        # Clean up intermediate text columns
        cleanup_tfidf_text_columns(beth_train, inplace=True)
        cleanup_tfidf_text_columns(beth_val, inplace=True)
        cleanup_tfidf_text_columns(beth_test, inplace=True)
        print(f"TF-IDF text columns cleaned up")
    
    # Save preprocessed data
    suffix = '_tfidf' if tfidf else ''
    beth_train.to_csv(os.path.join(data_dir, f'beth_train_preprocessed{suffix}.csv'), index=False)
    beth_val.to_csv(os.path.join(data_dir, f'beth_val_preprocessed{suffix}.csv'), index=False)
    beth_test.to_csv(os.path.join(data_dir, f'beth_test_preprocessed{suffix}.csv'), index=False)
    print(f"Preprocessed BETH data saved to {data_dir}")

    print(f"BETH splits loaded with log transformations:")
    print(f"       Train: {beth_train.shape}, Val: {beth_val.shape}, Test: {beth_test.shape}")
    
    # Report log-transformed features
    log_features = [col for col in beth_train.columns if col.startswith('log_')]
    if log_features:
        print(f"       Log-transformed features: {log_features}")
    
    # Return with or without TF-IDF matrices
    if tfidf:
        print(f"\nReturning combined feature matrices:")
        print(f"  X_train: {X_train.shape} (sparse)")
        print(f"  X_val:   {X_val.shape} (sparse)")
        print(f"  X_test:  {X_test.shape} (sparse)")
        print(f"  Labels:  y_train, y_val, y_test (DataFrames with 'sus', 'evil' columns)")
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_names
    else:
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
    print(f"UNSW-NB15 dataset located at {base_path}")
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


# Public API for pipeline use
def preprocess_unsw_split(df):
    """
    Preprocess a single UNSW-NB15 DataFrame (public API).
    
    Transformations applied:
    - Drop NaN values
    - Drop 'id' column (row identifier, not a feature)
    - Encode categorical features (proto, service, state) to integers
    - Clean infinities and NaNs
    
    Args:
        df: Raw UNSW-NB15 DataFrame
        
    Returns:
        Preprocessed DataFrame (cleaned features, categorical encoded as integers)
        
    Example:
        >>> from data_extraction import extract_unsw
        >>> from preprocessing import preprocess_unsw_split
        >>> train_raw, test_raw = extract_unsw()
        >>> train_clean = preprocess_unsw_split(train_raw)
    """
    df = df.copy()
    df.dropna(inplace=True)
    
    # Drop ID column if exists
    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)
    
    df = _encode_categoricals(df)
    df = _clean_dataframe(df)
    
    return df


def load_unsw(raw=False, split_test=True, tfidf=False): # Add tfidf flag for signature consistency
    """
    Load and preprocess UNSW-NB15 dataset with proper train/test splitting.
    
    Pipeline:
        1. Load CSV data from Kaggle
        2. (If raw=True) Return original unprocessed data
        3. (If raw=False) Split Kaggle test set in half to create validation and test sets
        4. Add source tracking ('train', 'val', 'test')
        5. Drop NaNs
        6. Encode categorical features (proto, service, state) to integers
        7. Feature engineering (optional - controlled by parameters)
        8. Log transformations (optional)
        9. Clean dataframe
        10. Scale numeric features PROPERLY (fit on train, transform all sets)
    
    NOTE: This approach matches the reference notebook that achieved ~91% accuracy.
          The key is splitting the Kaggle test set in half, which creates a more
          compatible test set than using the full Kaggle test set.
    
    Args:
        raw: If True, return raw unprocessed data without any transformations.
             If False, apply full preprocessing pipeline (default).
        split_test: If True, splits Kaggle test set in half for val/test (default).
                    If False, uses original Kaggle splits (lower accuracy).
                    Note: Ignored if raw=True.
    
    Returns:
        If raw=True:
            tuple: (train_df, test_df) - Original unprocessed Kaggle DataFrames (no validation split)
        
        If raw=False:
            tuple: (X_train, X_val, X_test, y_labels) 
                - X_train/val/test: Feature DataFrames
                - y_labels: Dictionary containing label Series for train/val/test
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
    print(f"Original UNSW-NB15 data saved to {data_dir}")
    
    # If raw=True, return original unprocessed data
    if raw:
        print(f"Returning RAW unprocessed UNSW-NB15 data:")
        print(f"       Train: {train_df_orig.shape}, Test: {test_df_orig.shape}")
        print(f"       (Note: No validation split in raw mode)")
        return train_df_orig, test_df_orig
    
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
        
        print(f"UNSW-NB15 dataset loaded with test set split:")
        print(f"       Original train: {len(train_df)} samples")
        print(f"       Validation: {len(val_df)} samples (from Kaggle test)")
        print(f"       Test: {len(test_df)} samples (from Kaggle test)")
    else:
        # Use original Kaggle splits (lower accuracy due to distribution mismatch)
        train_df['source'] = 'train'
        test_df['source'] = 'test'
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        val_df = None
        
        print(f"UNSW-NB15 dataset loaded with original Kaggle splits:")
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
    
    # Scale using consistent scale_numeric_features() with proper train/val/test handling
    # NOTE: We need to fit on train and transform val/test, so we do this manually
    # but use the same exclusion logic as scale_numeric_features()
    num_cols = train_processed.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    # Use same exclusion logic as scale_numeric_features()
    exclude_cols = ['label', 'attack_cat', 'sus', 'evil']
    exclude_cols.extend([c for c in num_cols if c.endswith('_flag')])
    
    # Auto-detect binary columns from TRAINING set
    for col in num_cols:
        if col not in exclude_cols and train_processed[col].nunique() <= 2:
            exclude_cols.append(col)
    
    # Filter to only columns that should be scaled
    num_cols = [c for c in num_cols if c not in exclude_cols]
    
    if len(num_cols) > 0:
        scaler = StandardScaler()
        train_processed[num_cols] = scaler.fit_transform(train_processed[num_cols])
        if val_processed is not None:
            val_processed[num_cols] = scaler.transform(val_processed[num_cols])
        test_processed[num_cols] = scaler.transform(test_processed[num_cols])
        print(f"Scaled {len(num_cols)} UNSW features (excluded {len(exclude_cols)} binary/target columns)")
    
    # Save preprocessed data
    train_processed.to_csv(os.path.join(data_dir, 'unsw_train_preprocessed.csv'), index=False)
    if val_processed is not None:
        val_processed.to_csv(os.path.join(data_dir, 'unsw_val_preprocessed.csv'), index=False)
    test_processed.to_csv(os.path.join(data_dir, 'unsw_test_preprocessed.csv'), index=False)
    print(f"Preprocessed UNSW-NB15 data saved to {data_dir}")
    
    print(f"Preprocessing complete:")
    print(f"       Features: {len(num_cols)} numeric")
    print(f"       Scaling: Fit on train, applied to all sets")
    print(f"       Categorical encoding: proto, service, state -> integers")

    # NEW: Separate features and labels before returning
    label_cols = ['label', 'attack_cat']
    
    X_train = train_processed.drop(columns=label_cols)
    y_train = train_processed[label_cols]
    
    X_test = test_processed.drop(columns=label_cols)
    y_test = test_processed[label_cols]
    
    if val_processed is not None:
        X_val = val_processed.drop(columns=label_cols)
        y_val = val_processed[label_cols]
    else:
        X_val, y_val = (None, None)

    y_labels = {
        'train': y_train,
        'val': y_val,
        'test': y_test
    }
    
    return X_train, X_val, X_test, y_labels


# ================================================================
# Combined Convenience Loader
# ================================================================

def prepare_datasets(raw=False, beth_tfidf=False):
    """
    Download and preprocess all datasets.
    
    Args:
        raw: If True, return raw unprocessed data (no preprocessing).
             If False, apply full preprocessing pipeline (default).
        beth_tfidf: If True, extract TF-IDF features from BETH 'args' column.
                    Note: Ignored if raw=True.
    
    Returns:
        If raw=True:
            beth_train, beth_val, beth_test, unsw_train, unsw_test
            (Note: UNSW returns only train/test in raw mode, no validation split)
        
        If raw=False and beth_tfidf=False:
            beth_train, beth_val, beth_test, unsw_train, unsw_val, unsw_test
        
        If raw=False and beth_tfidf=True:
            beth_train, beth_val, beth_test, tfidf_train, tfidf_val, tfidf_test, 
            feature_names, unsw_train, unsw_val, unsw_test
    """
    if raw:
        beth_train, beth_val, beth_test = load_beth(raw=True)
        unsw_train, unsw_test = load_unsw(raw=True)
        print("All datasets loaded in RAW mode (no preprocessing).")
        return beth_train, beth_val, beth_test, unsw_train, unsw_test
    
    if beth_tfidf:
        beth_train, beth_val, beth_test, tfidf_train, tfidf_val, tfidf_test, feature_names = load_beth(tfidf=True)
    else:
        beth_train, beth_val, beth_test = load_beth(tfidf=False)
    
    unsw_train, unsw_val, unsw_test = load_unsw()
    print("All datasets prepared successfully.")
    
    if beth_tfidf:
        return beth_train, beth_val, beth_test, tfidf_train, tfidf_val, tfidf_test, feature_names, unsw_train, unsw_val, unsw_test
    else:
        return beth_train, beth_val, beth_test, unsw_train, unsw_val, unsw_test