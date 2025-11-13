"""
data_extraction.py
------------------
Raw data extraction from KaggleHub (no transformations applied).

This module handles ONLY downloading and loading raw CSV files.
No preprocessing, feature engineering, or scaling is performed here.

Use this when you need:
- Raw data for EDA
- Custom preprocessing pipelines
- Comparison between raw and processed data
- Step-by-step pipeline development

For ready-to-model data, use data_loading.py instead.

Author: Joshua Laubach
Date: November 6, 2025
"""

import os
import pandas as pd
import kagglehub


# ================================================================
# BETH Dataset Extraction
# ================================================================

def extract_beth(save_to_disk=True):
    """
    Download and return raw BETH dataset from KaggleHub.
    
    NO preprocessing is applied - returns original CSV data as-is.
    
    Args:
        save_to_disk: If True, saves raw CSVs to data/beth/original/
        
    Returns:
        tuple: (train_df, val_df, test_df) - Raw unprocessed DataFrames
        
    Example:
        >>> train_raw, val_raw, test_raw = extract_beth()
        >>> print(train_raw.shape)  # Original columns and dtypes
    """
    print("\n" + "="*80)
    print("EXTRACTING BETH DATASET (RAW)")
    print("="*80)
    
    # Download from KaggleHub
    base_path = kagglehub.dataset_download("katehighnam/beth-dataset")
    print(f"Downloaded to: {base_path}")
    
    # Load raw CSVs
    train_df = pd.read_csv(os.path.join(base_path, "labelled_training_data.csv"))
    val_df = pd.read_csv(os.path.join(base_path, "labelled_validation_data.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "labelled_testing_data.csv"))
    
    print(f"\nRaw data loaded:")
    print(f"  Train:      {train_df.shape}")
    print(f"  Validation: {val_df.shape}")
    print(f"  Test:       {test_df.shape}")
    
    # Optionally save to project data directory
    if save_to_disk:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 'beth', 'original'
        )
        os.makedirs(data_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(data_dir, 'beth_train_raw.csv'), index=False)
        val_df.to_csv(os.path.join(data_dir, 'beth_val_raw.csv'), index=False)
        test_df.to_csv(os.path.join(data_dir, 'beth_test_raw.csv'), index=False)
        
        print(f"\nRaw CSVs saved to: {data_dir}")
    
    print("="*80 + "\n")
    
    return train_df, val_df, test_df


def get_beth_paths():
    """
    Get file paths to BETH dataset without loading data.
    
    Returns:
        dict: {'train': path, 'val': path, 'test': path}
    """
    base_path = kagglehub.dataset_download("katehighnam/beth-dataset")
    return {
        "train": os.path.join(base_path, "labelled_training_data.csv"),
        "val": os.path.join(base_path, "labelled_validation_data.csv"),
        "test": os.path.join(base_path, "labelled_testing_data.csv"),
    }


# ================================================================
# UNSW-NB15 Dataset Extraction
# ================================================================

def extract_unsw(save_to_disk=True):
    """
    Download and return raw UNSW-NB15 dataset from KaggleHub.
    
    NO preprocessing is applied - returns original Kaggle CSV data as-is.
    Note: Original Kaggle split is train/test only (no validation).
    
    Args:
        save_to_disk: If True, saves raw CSVs to data/unsw/original/
        
    Returns:
        tuple: (train_df, test_df) - Raw unprocessed DataFrames
        
    Example:
        >>> train_raw, test_raw = extract_unsw()
        >>> print(train_raw['attack_cat'].value_counts())  # Original labels
    """
    print("\n" + "="*80)
    print("EXTRACTING UNSW-NB15 DATASET (RAW)")
    print("="*80)
    
    # Download from KaggleHub
    base_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    print(f"Downloaded to: {base_path}")
    
    # Load raw CSVs
    train_df = pd.read_csv(os.path.join(base_path, "UNSW_NB15_training-set.csv"))
    test_df = pd.read_csv(os.path.join(base_path, "UNSW_NB15_testing-set.csv"))
    
    print(f"\nRaw data loaded:")
    print(f"  Train: {train_df.shape}")
    print(f"  Test:  {test_df.shape}")
    print(f"  (Note: Original Kaggle split has no validation set)")
    
    # Optionally save to project data directory
    if save_to_disk:
        data_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 
            'data', 'unsw', 'original'
        )
        os.makedirs(data_dir, exist_ok=True)
        
        train_df.to_csv(os.path.join(data_dir, 'unsw_train_raw.csv'), index=False)
        test_df.to_csv(os.path.join(data_dir, 'unsw_test_raw.csv'), index=False)
        
        print(f"\nRaw CSVs saved to: {data_dir}")
    
    print("="*80 + "\n")
    
    return train_df, test_df


def get_unsw_paths():
    """
    Get file paths to UNSW-NB15 dataset without loading data.
    
    Returns:
        dict: {'train': path, 'test': path}
    """
    base_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    return {
        "train": os.path.join(base_path, "UNSW_NB15_training-set.csv"),
        "test": os.path.join(base_path, "UNSW_NB15_testing-set.csv"),
    }


# ================================================================
# Utility Functions
# ================================================================

def extract_all_datasets(save_to_disk=True):
    """
    Extract all datasets (BETH + UNSW-NB15) in one call.
    
    Returns:
        dict: {
            'beth': (train, val, test),
            'unsw': (train, test)
        }
    """
    print("\n" + "="*80)
    print("EXTRACTING ALL DATASETS")
    print("="*80 + "\n")
    
    beth_data = extract_beth(save_to_disk=save_to_disk)
    unsw_data = extract_unsw(save_to_disk=save_to_disk)
    
    print("All datasets extracted successfully!")
    
    return {
        'beth': beth_data,
        'unsw': unsw_data
    }


if __name__ == "__main__":
    # Test extraction
    print("Testing data extraction...")
    
    beth_train, beth_val, beth_test = extract_beth()
    print(f"[OK] BETH extracted: {beth_train.shape}, {beth_val.shape}, {beth_test.shape}")
    
    unsw_train, unsw_test = extract_unsw()
    print(f"[OK] UNSW extracted: {unsw_train.shape}, {unsw_test.shape}")
