"""
data_loading.py
---------------
High-level data loading functions for immediate modeling.

ONE-CALL convenience functions that return fully transformed, ready-to-model data.
Applies complete pipeline: extraction -> preprocessing -> feature engineering -> scaling.

Use this in notebooks when you want to:
- Start modeling immediately
- Skip manual pipeline steps
- Use standardized transformations
- Focus on model development

For step-by-step pipeline control, use:
- data_extraction.py -> Raw data
- preprocessing.py -> Clean and transform
- feature_engineering.py -> Create features
- feature_selection.py -> Select best features

Author: Joshua Laubach
Date: November 6, 2025
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import pipeline components
from data_extraction import extract_beth, extract_unsw
from preprocessing import (
    preprocess_beth_split, 
    preprocess_unsw_split,
    scale_numeric_features
)
from feature_engineering import (
    engineer_beth_tfidf_features,
    cleanup_tfidf_text_columns,
    _apply_log_transforms_beth,
    _apply_log_transforms_unsw,
    _feature_engineer_unsw
)


# ================================================================
# BETH Dataset - Ready-to-Model Loading
# ================================================================

def load_beth(tfidf=False, max_features=500, min_df=2, max_df=0.5, 
              save_to_disk=True, verbose=True):
    """
    Load fully preprocessed BETH dataset - READY FOR IMMEDIATE MODELING.
    
    Complete pipeline applied:
    1. Extract raw data from KaggleHub
    2. Preprocess (clean, encode, create flag features)
    3. Apply log transformations to skewed features
    4. (Optional) Extract TF-IDF features from 'args' text
    5. Scale numeric features
    6. Return ready-to-model DataFrames
    
    Args:
        tfidf: If True, extract TF-IDF features from 'args' column (default: False)
        max_features: Max TF-IDF features (only used if tfidf=True)
        min_df: Min document frequency for TF-IDF (only used if tfidf=True)
        max_df: Max document frequency for TF-IDF (only used if tfidf=True)
        save_to_disk: If True, saves processed data to data/beth/processed/
        verbose: Print progress information
        
    Returns:
        If tfidf=False:
            tuple: (train_df, val_df, test_df)
                - DataFrames with preprocessed numeric features
                - Columns: numeric features + 'sus' + 'evil' labels
                
        If tfidf=True:
            tuple: (train_df, val_df, test_df, tfidf_train, tfidf_val, tfidf_test, feature_names)
                - train/val/test_df: Preprocessed numeric features (text columns removed)
                - tfidf_train/val/test: Sparse TF-IDF feature matrices
                - feature_names: List of TF-IDF feature names
                
    Example (Immediate Modeling):
        >>> from data_loading import load_beth
        >>> from models_unsupervised import KMeansAnomalyDetector
        >>> 
        >>> # One line to get ready-to-model data
        >>> train, val, test = load_beth(tfidf=False)
        >>> 
        >>> # Separate features and labels
        >>> X_train = train.drop(['sus', 'evil'], axis=1)
        >>> y_sus = train['sus']
        >>> 
        >>> # Start modeling immediately!
        >>> model = KMeansAnomalyDetector(n_clusters=8)
        >>> model.fit(X_train)
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING BETH DATASET (FULLY PREPROCESSED)")
        print("="*80)
    
    # Step 1: Extract raw data
    if verbose:
        print("\n[1/5] Extracting raw data from KaggleHub...")
    train_raw, val_raw, test_raw = extract_beth(save_to_disk=False)
    
    # Step 2: Preprocess each split
    if verbose:
        print("[2/5] Preprocessing (cleaning, encoding, flag features)...")
    train = preprocess_beth_split(train_raw)
    val = preprocess_beth_split(val_raw)
    test = preprocess_beth_split(test_raw)
    
    # Step 3: Apply log transformations
    if verbose:
        print("[3/5] Applying log transformations to skewed features...")
    train = _apply_log_transforms_beth(train)
    val = _apply_log_transforms_beth(val)
    test = _apply_log_transforms_beth(test)
    
    # Step 4: TF-IDF feature extraction (optional)
    if tfidf:
        if verbose:
            print(f"[4/5] Extracting TF-IDF features (max={max_features}, min_df={min_df}, max_df={max_df})...")
        
        tfidf_train, tfidf_val, tfidf_test, feature_names, vectorizer = engineer_beth_tfidf_features(
            train, val, test,
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            verbose=verbose
        )
        
        # Clean up intermediate text columns
        cleanup_tfidf_text_columns(train, inplace=True)
        cleanup_tfidf_text_columns(val, inplace=True)
        cleanup_tfidf_text_columns(test, inplace=True)
    else:
        if verbose:
            print("[4/5] Skipping TF-IDF extraction (tfidf=False)")
    
    # Step 5: Scale numeric features
    if verbose:
        print("[5/5] Scaling numeric features...")
    train = scale_numeric_features(train)
    val = scale_numeric_features(val)
    test = scale_numeric_features(test)
    
    # Save processed data to disk
    if save_to_disk:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 'beth', 'processed'
        )
        os.makedirs(data_dir, exist_ok=True)
        
        suffix = '_tfidf' if tfidf else ''
        train.to_csv(os.path.join(data_dir, f'beth_train{suffix}.csv'), index=False)
        val.to_csv(os.path.join(data_dir, f'beth_val{suffix}.csv'), index=False)
        test.to_csv(os.path.join(data_dir, f'beth_test{suffix}.csv'), index=False)
        
        if verbose:
            print(f"\nProcessed data saved to: {data_dir}")
    
    if verbose:
        print("\n" + "="*80)
        print("[OK] BETH DATASET READY FOR MODELING")
        print("="*80)
        print(f"Train:      {train.shape}")
        print(f"Validation: {val.shape}")
        print(f"Test:       {test.shape}")
        if tfidf:
            print(f"\nTF-IDF Features: {tfidf_train.shape[1]} text features extracted")
        print(f"Labels: 'sus' (in-distribution outliers), 'evil' (out-of-distribution outliers)")
        print("="*80 + "\n")
    
    # Return with or without TF-IDF matrices
    if tfidf:
        return train, val, test, tfidf_train, tfidf_val, tfidf_test, feature_names
    else:
        return train, val, test


# ================================================================
# UNSW-NB15 Dataset - Ready-to-Model Loading
# ================================================================

def load_unsw(split_test=True, save_to_disk=True, verbose=True):
    """
    Load fully preprocessed UNSW-NB15 dataset - READY FOR IMMEDIATE MODELING.
    
    Complete pipeline applied:
    1. Extract raw data from KaggleHub
    2. (Optional) Split test set in half to create validation set
    3. Preprocess (clean, encode categoricals, drop NaNs)
    4. Feature engineering (source-destination pairs, derived features)
    5. Apply log transformations to skewed features
    6. Scale numeric features (fit on train, transform val/test)
    7. Return ready-to-model DataFrames
    
    Args:
        split_test: If True, splits Kaggle test in half for val/test (recommended)
                    If False, uses original Kaggle train/test split (no validation)
        save_to_disk: If True, saves processed data to data/unsw/processed/
        verbose: Print progress information
        
    Returns:
        If split_test=True:
            tuple: (train_df, val_df, test_df)
                - DataFrames with preprocessed features
                - Columns: numeric features + 'label' (binary) + 'attack_cat' (multi-class)
                
        If split_test=False:
            tuple: (train_df, None, test_df)
                - No validation split
                
    Example (Immediate Modeling):
        >>> from data_loading import load_unsw
        >>> from models_supervised import XGBoostClassifier
        >>> 
        >>> # One line to get ready-to-model data
        >>> train, val, test = load_unsw(split_test=True)
        >>> 
        >>> # Separate features and labels
        >>> X_train = train.drop(['label', 'attack_cat'], axis=1)
        >>> y_train = train['label']
        >>> 
        >>> X_val = val.drop(['label', 'attack_cat'], axis=1)
        >>> y_val = val['label']
        >>> 
        >>> # Start modeling immediately!
        >>> model = XGBoostClassifier()
        >>> model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING UNSW-NB15 DATASET (FULLY PREPROCESSED)")
        print("="*80)
    
    # Step 1: Extract raw data
    if verbose:
        print("\n[1/6] Extracting raw data from KaggleHub...")
    train_raw, test_raw = extract_unsw(save_to_disk=False)
    
    # Step 2: Split test set for validation (optional but recommended)
    if split_test:
        if verbose:
            print("[2/6] Splitting test set in half to create validation set...")
        test_raw, val_raw = train_test_split(
            test_raw, 
            test_size=0.5, 
            random_state=42, 
            stratify=test_raw['label']
        )
        if verbose:
            print(f"       Train: {len(train_raw):,} | Val: {len(val_raw):,} | Test: {len(test_raw):,}")
    else:
        val_raw = None
        if verbose:
            print("[2/6] Using original Kaggle splits (no validation set)")
    
    # Step 3: Preprocess each split
    if verbose:
        print("[3/6] Preprocessing (cleaning, encoding categoricals, dropping NaNs)...")
    train = preprocess_unsw_split(train_raw)
    val = preprocess_unsw_split(val_raw) if val_raw is not None else None
    test = preprocess_unsw_split(test_raw)
    
    # Step 4: Feature engineering
    if verbose:
        print("[4/6] Feature engineering (source-destination pairs, derived features)...")
    train = _feature_engineer_unsw(train)
    if val is not None:
        val = _feature_engineer_unsw(val)
    test = _feature_engineer_unsw(test)
    
    # Step 5: Apply log transformations
    if verbose:
        print("[5/6] Applying log transformations to skewed features...")
    train = _apply_log_transforms_unsw(train)
    if val is not None:
        val = _apply_log_transforms_unsw(val)
    test = _apply_log_transforms_unsw(test)
    
    # Step 6: Scale numeric features (fit on train, transform others)
    if verbose:
        print("[6/6] Scaling numeric features (fit on train, transform val/test)...")
    
    # Manual scaling with proper fit/transform separation
    from sklearn.preprocessing import StandardScaler
    
    # Get numeric columns (excluding targets and binary flags)
    num_cols = train.select_dtypes(include=["float64", "int64"]).columns.tolist()
    exclude_cols = ['label', 'attack_cat']
    exclude_cols.extend([c for c in num_cols if c.endswith('_flag')])
    
    # Auto-detect binary columns from training set
    for col in num_cols:
        if col not in exclude_cols and train[col].nunique() <= 2:
            exclude_cols.append(col)
    
    # Filter to only columns that should be scaled
    scale_cols = [c for c in num_cols if c not in exclude_cols]
    
    if len(scale_cols) > 0:
        scaler = StandardScaler()
        train[scale_cols] = scaler.fit_transform(train[scale_cols])
        if val is not None:
            val[scale_cols] = scaler.transform(val[scale_cols])
        test[scale_cols] = scaler.transform(test[scale_cols])
        
        if verbose:
            print(f"       Scaled {len(scale_cols)} features (excluded {len(exclude_cols)} binary/target columns)")
    
    # Save processed data to disk
    if save_to_disk:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 'unsw', 'processed'
        )
        os.makedirs(data_dir, exist_ok=True)
        
        train.to_csv(os.path.join(data_dir, 'unsw_train.csv'), index=False)
        if val is not None:
            val.to_csv(os.path.join(data_dir, 'unsw_val.csv'), index=False)
        test.to_csv(os.path.join(data_dir, 'unsw_test.csv'), index=False)
        
        if verbose:
            print(f"\nProcessed data saved to: {data_dir}")
    
    if verbose:
        print("\n" + "="*80)
        print("[OK] UNSW-NB15 DATASET READY FOR MODELING")
        print("="*80)
        print(f"Train:      {train.shape}")
        if val is not None:
            print(f"Validation: {val.shape}")
        print(f"Test:       {test.shape}")
        print(f"\nLabels: 'label' (binary: 0=normal, 1=attack), 'attack_cat' (multi-class attack type)")
        print("="*80 + "\n")
    
    return train, val, test


# ================================================================
# Load All Datasets
# ================================================================

def load_all_datasets(beth_tfidf=False, unsw_split_test=True, verbose=True):
    """
    Load both BETH and UNSW-NB15 datasets in one call.
    
    Returns:
        dict: {
            'beth': (train, val, test) or (train, val, test, tfidf_train, tfidf_val, tfidf_test, names),
            'unsw': (train, val, test)
        }
    """
    if verbose:
        print("\n" + "="*80)
        print("LOADING ALL DATASETS")
        print("="*80 + "\n")
    
    beth_data = load_beth(tfidf=beth_tfidf, verbose=verbose)
    unsw_data = load_unsw(split_test=unsw_split_test, verbose=verbose)
    
    if verbose:
        print("\n[OK] All datasets loaded and ready for modeling!\n")
    
    return {
        'beth': beth_data,
        'unsw': unsw_data
    }


if __name__ == "__main__":
    # Test loading
    print("Testing data loading pipeline...")
    
    # Test BETH
    print("\n" + "="*80)
    print("Testing BETH loading...")
    beth_train, beth_val, beth_test = load_beth(tfidf=False)
    print(f"[OK] BETH loaded: Train {beth_train.shape}, Val {beth_val.shape}, Test {beth_test.shape}")
    print(f"  Feature columns: {[c for c in beth_train.columns if c not in ['sus', 'evil']][:5]}...")
    print(f"  Target columns: {[c for c in beth_train.columns if c in ['sus', 'evil']]}")
    
    # Test UNSW
    print("\n" + "="*80)
    print("Testing UNSW loading...")
    unsw_train, unsw_val, unsw_test = load_unsw(split_test=True)
    print(f"[OK] UNSW loaded: Train {unsw_train.shape}, Val {unsw_val.shape}, Test {unsw_test.shape}")
    print(f"  Feature columns: {[c for c in unsw_train.columns if c not in ['label', 'attack_cat']][:5]}...")
    print(f"  Target columns: {[c for c in unsw_train.columns if c in ['label', 'attack_cat']]}")
