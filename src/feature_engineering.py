"""
feature_engineering.py
----------------------
Advanced feature engineering for network security datasets.

Includes:
  - BETH: TF-IDF text features from system call arguments
  - UNSW-NB15: Source-destination pair features (sum, diff, ratio, zero indicators)
"""

import ast
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import kagglehub


# ================================================================
# ----------------- Utility Functions -----------------------------
# ================================================================

def scale_numeric_features(df, exclude_cols=None, max_unique_categorical=10, auto_detect=True):
    """
    Scale continuous numeric columns using StandardScaler.
    
    IMPORTANT: Does NOT scale binary/categorical features to preserve interpretability.
    
    Args:
        df: DataFrame with features
        exclude_cols: Explicit list of column names to exclude. If None, uses auto-detection.
        max_unique_categorical: Maximum unique values to consider a feature categorical (default: 10)
        auto_detect: If True, automatically detect binary/categorical features based on unique value counts
        
    Returns:
        DataFrame with scaled continuous features, unscaled binary/categorical features
        
    Detection Logic:
        - Binary: <= 2 unique values (e.g., 0/1 flags, True/False)
        - Categorical: <= max_unique_categorical unique values (e.g., proto, state codes)
        - Continuous: > max_unique_categorical unique values
    """
    # Get all numeric columns
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    
    # Auto-detect binary/categorical features if not explicitly provided
    if exclude_cols is None and auto_detect:
        exclude_cols = []
        detection_details = []
        
        for col in num_cols:
            n_unique = df[col].nunique()
            
            if n_unique <= 2:
                exclude_cols.append(col)
                detection_details.append((col, n_unique, 'binary'))
            elif n_unique <= max_unique_categorical:
                exclude_cols.append(col)
                detection_details.append((col, n_unique, 'categorical'))
        
        if detection_details:
            print(f"Auto-detected {len(exclude_cols)} binary/categorical features:")
            for col, n_unique, feat_type in detection_details[:10]:  # Show first 10
                print(f"   - {col:30} ({n_unique} unique values, {feat_type})")
            if len(detection_details) > 10:
                print(f"   ... and {len(detection_details) - 10} more")
    
    elif exclude_cols is None:
        # No exclusions provided and auto-detect disabled
        exclude_cols = []
        print(f"[WARN] No exclusions provided and auto_detect=False. All numeric features will be scaled.")
    
    # Separate columns to scale vs exclude
    cols_to_scale = [col for col in num_cols if col not in exclude_cols]
    cols_excluded = [col for col in num_cols if col in exclude_cols]
    
    # Scale only continuous features
    if len(cols_to_scale) > 0:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        print(f"Scaled {len(cols_to_scale)} continuous features")
        if cols_excluded:
            print(f"Excluded {len(cols_excluded)} binary/categorical features from scaling")
    else:
        print(f"[WARN] No continuous features found to scale")
    
    return df


def _safe_drop(df, columns):
    """Drop columns safely without errors."""
    df.drop(columns=[c for c in columns if c in df.columns], inplace=True, errors="ignore")
    return df


def _clean_dataframe(df):
    """Replace infinities and NaNs."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    return df


# ================================================================
# ------------------ BETH DATASET --------------------------------
# ================================================================

def _get_beth_paths():
    """Download and return file paths for BETH dataset."""
    base_path = kagglehub.dataset_download("katehighnam/beth-dataset")
    paths = {
        "train": os.path.join(base_path, "labelled_training_data.csv"),
        "val": os.path.join(base_path, "labelled_validation_data.csv"),
        "test": os.path.join(base_path, "labelled_testing_data.csv"),
    }
    print(f"BETH dataset path: {base_path}")
    return paths


def _preprocess_beth_split(df):
    """Apply BETH-specific preprocessing and feature encoding."""
    df.rename(columns={"timestamp": "timeSinceBoot"}, inplace=True)
    df["processId_flag"] = df["processId"].isin([0, 1, 2]).astype(int)
    df["parentProcessId_flag"] = df["parentProcessId"].isin([0, 1, 2]).astype(int)
    df["userId_flag"] = (df["userId"] < 1000).astype(int)
    df["mountNamespace_flag"] = (df["mountNamespace"] == 4026531840).astype(int)
    df["returnValue_flag"] = np.sign(df["returnValue"]).astype(int)

    _safe_drop(df, ["threadId", "processName", "stackAddresses", "eventName"])
    if "hostName" in df.columns:
        df["hostName"] = df["hostName"].astype("category").cat.codes

    return _clean_dataframe(df)


def _apply_log_transforms_beth(df):
    """
    Apply log1p transformation to highly skewed BETH features.
    
    Transforms:
        - timeSinceBoot -> log_timeSinceBoot (large timestamp values)
        - processId -> log_processId (can be very large)
        - parentProcessId -> log_parentProcessId (can be very large)
        
    Creates new log-transformed columns and removes originals to avoid multicollinearity.
    """
    print(f"Applying log1p transformations to BETH skewed features...")
    
    skewed_features = ['timeSinceBoot', 'processId', 'parentProcessId']
    transformed = []
    
    for feat in skewed_features:
        if feat in df.columns:
            # Create new log-transformed column
            log_col = f'log_{feat}'
            df[log_col] = np.log1p(df[feat])
            transformed.append(log_col)
            
            # Remove original to avoid multicollinearity
            df.drop(columns=[feat], inplace=True)
    
    print(f"Created {len(transformed)} log-transformed features:")
    for feat in transformed:
        print(f"   - {feat} (original removed)")
    
    return df


def load_beth():
    """Load and preprocess all three BETH splits."""
    paths = _get_beth_paths()
    beth_train = pd.read_csv(paths["train"])
    beth_val = pd.read_csv(paths["val"])
    beth_test = pd.read_csv(paths["test"])

    for df in [beth_train, beth_val, beth_test]:
        df = _preprocess_beth_split(df)
        df = _apply_log_transforms_beth(df)    # Apply log transforms
        df = scale_numeric_features(df)        # Auto-detect and standardize

    print(f"BETH splits loaded: Train {beth_train.shape}, Val {beth_val.shape}, Test {beth_test.shape}")
    return beth_train, beth_val, beth_test


# ================================================================
# ------------------ BETH TEXT FEATURES --------------------------
# ================================================================

def _parse_args_column(df, args_col='args'):
    """
    Parse the 'args' JSON column and extract structured argument information.
    
    Args:
        df: DataFrame containing args column
        args_col: Name of the column containing JSON argument data
        
    Returns:
        DataFrame with additional columns: arg_names, arg_types, arg_values, args_str
    """
    print(f"Parsing '{args_col}' column...")
    
    # Parse JSON strings
    try:
        df[args_col] = df[args_col].apply(ast.literal_eval)
    except:
        print(f"[WARN] Args column may already be parsed or in different format")
    
    # Extract argument components
    df['arg_names'] = df[args_col].apply(
        lambda x: [d['name'] for d in x if isinstance(d, dict)] if isinstance(x, list) else []
    )
    df['arg_types'] = df[args_col].apply(
        lambda x: [d['type'] for d in x if isinstance(d, dict)] if isinstance(x, list) else []
    )
    df['arg_values'] = df[args_col].apply(
        lambda x: [str(d['value']) for d in x if isinstance(d, dict)] if isinstance(x, list) else []
    )
    
    # Create structured argument strings: "name=value(type)"
    df['args_str'] = df.apply(
        lambda row: '|||'.join(
            f"{name}={value}({type_})" 
            for name, value, type_ in zip(row['arg_names'], row['arg_values'], row['arg_types'])
        ) if len(row['arg_names']) > 0 else '', 
        axis=1
    )
    
    print(f"Parsed {len(df)} records into structured argument strings")
    return df


def _custom_tokenizer(text):
    """
    Custom tokenizer for system call argument patterns.
    Splits on ||| delimiter instead of whitespace.
    
    Args:
        text: String containing argument patterns separated by |||
        
    Returns:
        List of argument pattern tokens
    """
    return text.split('|||') if text else []


def engineer_beth_tfidf_features(df_train, df_val, df_test, 
                                  max_features=500, 
                                  min_df=2, 
                                  max_df=0.5,
                                  verbose=True):
    """
    Apply TF-IDF vectorization to BETH dataset system call arguments.
    
    This function:
    1. Parses the 'args' JSON column into structured text
    2. Applies TF-IDF vectorization to capture argument patterns
    3. Returns sparse TF-IDF feature matrices for train/val/test
    
    Args:
        df_train: Training DataFrame with 'args' column
        df_val: Validation DataFrame with 'args' column  
        df_test: Test DataFrame with 'args' column
        max_features: Maximum number of TF-IDF features to generate
        min_df: Minimum document frequency (ignore rare patterns)
        max_df: Maximum document frequency (ignore common patterns)
        verbose: Print progress information
        
    Returns:
        tuple: (tfidf_train, tfidf_val, tfidf_test, feature_names, vectorizer)
            - tfidf_train/val/test: Sparse CSR matrices of TF-IDF features
            - feature_names: List of TF-IDF feature names
            - vectorizer: Fitted TfidfVectorizer object
    """
    if verbose:
        print("\n" + "="*80)
        print("BETH TF-IDF TEXT FEATURE ENGINEERING")
        print("="*80)
    
    # Parse args column for all splits
    df_train = _parse_args_column(df_train.copy())
    df_val = _parse_args_column(df_val.copy())
    df_test = _parse_args_column(df_test.copy())
    
    # Initialize TF-IDF vectorizer
    if verbose:
        print(f"\nInitializing TF-IDF vectorizer...")
        print(f"   - max_features: {max_features}")
        print(f"   - min_df: {min_df} (ignore patterns in < {min_df} documents)")
        print(f"   - max_df: {max_df} (ignore patterns in > {max_df*100}% of documents)")
    
    vectorizer = TfidfVectorizer(
        tokenizer=_custom_tokenizer,
        token_pattern=None,        # Disable regex tokenization
        lowercase=False,           # Preserve case for system calls
        max_features=max_features,
        min_df=min_df,
        max_df=max_df
    )
    
    # Fit on training data and transform all splits
    if verbose:
        print(f"\nFitting TF-IDF on training data...")
    
    tfidf_train = vectorizer.fit_transform(df_train['args_str'])
    tfidf_val = vectorizer.transform(df_val['args_str'])
    tfidf_test = vectorizer.transform(df_test['args_str'])
    
    feature_names = vectorizer.get_feature_names_out()
    
    if verbose:
        print(f"TF-IDF feature extraction complete!")
        print(f"\n[TF-IDF Matrix Shapes]")
        print(f"   Training:   {tfidf_train.shape}")
        print(f"   Validation: {tfidf_val.shape}")
        print(f"   Test:       {tfidf_test.shape}")
        print(f"\n[TF-IDF Statistics]")
        print(f"   Unique argument patterns captured: {len(feature_names)}")
        print(f"   Sparsity (train): {100 * (1 - tfidf_train.nnz / (tfidf_train.shape[0] * tfidf_train.shape[1])):.2f}%")
        print(f"   Memory (train): {tfidf_train.data.nbytes / 1024**2:.2f} MB (sparse)")
        
        # Show sample features
        print(f"\n[Sample TF-IDF Features]")
        for i, feat in enumerate(feature_names[:10], 1):
            print(f"   {i:2d}. {feat}")
        if len(feature_names) > 10:
            print(f"   ... and {len(feature_names) - 10} more")
    
    print("="*80 + "\n")
    
    # BUG FIX: Remove intermediate text columns after TF-IDF extraction
    # These columns should NOT be left in the original DataFrames
    # Note: This modifies the INPUT dataframes (df_train, df_val, df_test)
    # If you need to preserve originals, pass copies to this function
    text_cols_to_remove = ['args', 'arg_names', 'arg_types', 'arg_values', 'args_str']
    
    # Clean up is handled by the caller, so we just return the matrices
    # The notebook/calling code should drop these columns from the original DataFrames
    
    return tfidf_train, tfidf_val, tfidf_test, feature_names, vectorizer


def combine_numeric_and_tfidf(df, tfidf_matrix, exclude_cols=None):
    """
    Combine numeric features with TF-IDF sparse matrix.
    
    Args:
        df: DataFrame with numeric features
        tfidf_matrix: Sparse TF-IDF feature matrix
        exclude_cols: List of columns to exclude (e.g., target labels, text columns)
        
    Returns:
        Sparse matrix combining numeric and TF-IDF features
    """
    if exclude_cols is None:
        exclude_cols = ['sus', 'evil', 'args', 'arg_names', 'arg_types', 'arg_values', 'args_str']
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Convert numeric features to sparse matrix
    X_numeric = csr_matrix(df[numeric_cols].values)
    
    # Combine with TF-IDF features
    X_combined = hstack([X_numeric, tfidf_matrix])
    
    print(f"Combined features: {X_numeric.shape[1]} numeric + {tfidf_matrix.shape[1]} TF-IDF = {X_combined.shape[1]} total")
    
    return X_combined


# ================================================================
# ------------------ UNSW-NB15 DATASET ----------------------------
# ================================================================

def _get_unsw_paths():
    """Download and return file paths for UNSW-NB15 dataset."""
    base_path = kagglehub.dataset_download("mrwellsdavid/unsw-nb15")
    paths = {
        "train": os.path.join(base_path, "UNSW_NB15_training-set.csv"),
        "test": os.path.join(base_path, "UNSW_NB15_testing-set.csv"),
    }
    print(f"UNSW-NB15 dataset path: {base_path}")
    return paths


def identify_src_dst_pairs(df):
    """
    Identify all source-destination column pairs dynamically.
    Returns a dictionary of {source_col: destination_col}.
    """
    src_dst_pairs = {}

    for col in df.columns:
        # Case 1: simple symmetric prefix (spkts/dpkts)
        if col.startswith("s") and "d" + col[1:] in df.columns:
            src_dst_pairs[col] = "d" + col[1:]
        # Case 2: src_*port <-> dst_*port (swapped positions)
        # e.g., ct_src_dport_ltm <-> ct_dst_sport_ltm
        # Check this BEFORE general src/dst replacement
        elif "src_dport" in col:
            d_pair = col.replace("src_dport", "dst_sport")
            if d_pair in df.columns:
                src_dst_pairs[col] = d_pair
        elif "src_sport" in col:
            d_pair = col.replace("src_sport", "dst_dport")
            if d_pair in df.columns:
                src_dst_pairs[col] = d_pair
        # Case 3: general src/dst naming pattern
        elif "src" in col:
            d_pair = col.replace("src", "dst")
            if d_pair in df.columns:
                src_dst_pairs[col] = d_pair

    print(f"Found {len(src_dst_pairs)} source-destination pairs.")
    for s, d in src_dst_pairs.items():
        print(f"   {s} <-> {d}")
    return src_dst_pairs


def _feature_engineer_unsw(df):
    """
    Comprehensive feature engineering for UNSW-NB15 dataset.
    
    Automatically generates derived numeric features for each detected source-destination 
    column pair, plus advanced interaction and extreme value features based on EDA insights.

    Features created per src-dst pair:
      - sum: s + d
      - diff: s - d
      - ratio: s / (d + 1)
      - both_zero: both values zero
      - one_zero: exactly one value zero
      - is_integer_ratio: ratio is close to an integer (within tolerance)
      - distance: normalized asymmetry distance from diagonal (y=x)
      - log_distance: asymmetry in log space
    
    Additional advanced features:
      - Extreme value indicators (tiny packets, extreme jitter, unidirectional loss)
      - Throughput efficiency (bytes per time unit)
      - Interaction features (temporal x spatial combinations)
      - Composite asymmetry score (combines top discriminative pairs)
      - Coefficient of variation (relative variability between src-dst pairs)
    
    Total features created: ~30-35 features per pair + 15 global features
    Feature selection will reduce this to top 30 most important features.
    """
    src_dst_pairs = identify_src_dst_pairs(df)
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE UNSW-NB15 FEATURE ENGINEERING")
    print(f"{'='*80}")
    print(f"Detected {len(src_dst_pairs)} source-destination pairs")
    print(f"Creating derived features for each pair...")

    # ============================================================================
    # PART 1: Source-Destination Pair Features
    # ============================================================================
    for src, dst in src_dst_pairs.items():
        try:
            # Small epsilon to avoid division by zero
            epsilon = 1e-6
            
            # Basic derived features
            df[f"sum_{src}_{dst}"] = df[src] + df[dst]
            df[f"diff_{src}_{dst}"] = df[src] - df[dst]
            df[f"ratio_{src}_{dst}"] = df[src] / (df[dst] + epsilon)
            df[f"both_zero_{src}_{dst}"] = ((df[src] == 0) & (df[dst] == 0)).astype(int)
            df[f"one_zero_{src}_{dst}"] = ((df[src] == 0) ^ (df[dst] == 0)).astype(int)
            
            # Integer ratio detection
            # Detects when ratio (or its reciprocal) is close to an integer
            # This captures gridded patterns visible in scatter plots
            tolerance = 0.05  # Tolerance for "close to integer"
            
            # Compute max ratio (handles both src/dst and dst/src)
            ratio_forward = df[src] / (df[dst] + epsilon)
            ratio_reverse = df[dst] / (df[src] + epsilon)
            max_ratio = np.maximum(ratio_forward, ratio_reverse)
            
            # Check if max ratio is close to an integer
            # Use round() instead of floor() to find nearest integer
            nearest_integer = np.round(max_ratio)
            distance_to_integer = np.abs(max_ratio - nearest_integer)
            
            # Binary feature: 1 if ratio is near-integer, 0 otherwise
            df[f"is_integer_ratio_{src}_{dst}"] = (distance_to_integer < tolerance).astype(int)
            
            # ASYMMETRY FEATURES (from scatter plot analysis)
            # Normalized distance from diagonal (y=x line)
            # Measures deviation from symmetric communication
            df[f"distance_{src}_{dst}"] = np.abs(df[src] - df[dst]) / (df[src] + df[dst] + epsilon)
            
            # Log-space distance (better for multiplicative relationships)
            df[f"log_distance_{src}_{dst}"] = np.abs(np.log1p(df[src]) - np.log1p(df[dst]))
            
        except Exception as e:
            print(f"[WARN] Could not engineer features for {src}-{dst}: {e}")

    print(f"[OK] Created {len(src_dst_pairs) * 8} pair-based features (8 per pair)")
    
    # ============================================================================
    # PART 2: Advanced Global Features (from EDA insights)
    # ============================================================================
    print(f"\nCreating advanced global features...")
    features_created = 0
    
    # 1. Extreme Value Indicators
    if 'smean' in df.columns and 'dmean' in df.columns:
        # Tiny packets (potential probes/scans) - log-transformed plots showed these at log~0
        df['is_tiny_src_pkt'] = (df['smean'] < 10).astype(int)
        df['is_tiny_dst_pkt'] = (df['dmean'] < 10).astype(int)
        df['has_tiny_packet'] = ((df['smean'] < 10) | (df['dmean'] < 10)).astype(int)
        features_created += 3
    
    if 'sjit' in df.columns and 'djit' in df.columns:
        # Extreme jitter (from visualization: attacks showed jitter up to log~14)
        df['is_extreme_src_jitter'] = (df['sjit'] > 1e6).astype(int)
        df['is_extreme_dst_jitter'] = (df['djit'] > 1e6).astype(int)
        df['has_extreme_jitter'] = ((df['sjit'] > 1e6) | (df['djit'] > 1e6)).astype(int)
        features_created += 3
    
    if 'sloss' in df.columns and 'dloss' in df.columns:
        # Unidirectional loss (vertical/horizontal bands in scatter plots)
        df['unidirectional_loss'] = (
            ((df['sloss'] == 0) & (df['dloss'] > 0)) | 
            ((df['dloss'] == 0) & (df['sloss'] > 0))
        ).astype(int)
        features_created += 1
    
    # 2. Throughput Efficiency
    if all(col in df.columns for col in ['sbytes', 'dbytes', 'sinpkt', 'dinpkt']):
        epsilon = 1e-6
        # Total bytes per inter-packet time (higher = more efficient)
        df['throughput_efficiency'] = (df['sbytes'] + df['dbytes']) / (df['sinpkt'] + df['dinpkt'] + epsilon)
        features_created += 1
    
    # 3. Interaction Features (Temporal x Spatial)
    # Most discriminative pairs from log-transformed visualizations
    if 'sinpkt' in df.columns and 'smean' in df.columns:
        df['ipt_size_product_src'] = df['sinpkt'] * df['smean']
        features_created += 1
    
    if 'dinpkt' in df.columns and 'dmean' in df.columns:
        df['ipt_size_product_dst'] = df['dinpkt'] * df['dmean']
        features_created += 1
    
    # 4. Composite Asymmetry Score
    # Combine the two most discriminative asymmetry measures
    asymmetry_components = []
    
    if 'distance_sinpkt_dinpkt' in df.columns:
        asymmetry_components.append(df['distance_sinpkt_dinpkt'])
    
    if 'distance_smean_dmean' in df.columns:
        asymmetry_components.append(df['distance_smean_dmean'])
    
    if len(asymmetry_components) >= 2:
        # Average of top 2 asymmetry measures
        df['asymmetry_score'] = sum(asymmetry_components) / len(asymmetry_components)
        features_created += 1
    
    # 5. Coefficient of Variation for paired features
    # Measures relative variability (std/mean) between src-dst pairs
    if 'sbytes' in df.columns and 'dbytes' in df.columns:
        epsilon = 1e-6
        mean_bytes = (df['sbytes'] + df['dbytes']) / 2
        std_bytes = np.sqrt(((df['sbytes'] - mean_bytes)**2 + (df['dbytes'] - mean_bytes)**2) / 2)
        df['cv_bytes'] = std_bytes / (mean_bytes + epsilon)
        features_created += 1
    
    if 'spkts' in df.columns and 'dpkts' in df.columns:
        epsilon = 1e-6
        mean_pkts = (df['spkts'] + df['dpkts']) / 2
        std_pkts = np.sqrt(((df['spkts'] - mean_pkts)**2 + (df['dpkts'] - mean_pkts)**2) / 2)
        df['cv_pkts'] = std_pkts / (mean_pkts + epsilon)
        features_created += 1
    
    if 'sinpkt' in df.columns and 'dinpkt' in df.columns:
        epsilon = 1e-6
        mean_ipt = (df['sinpkt'] + df['dinpkt']) / 2
        std_ipt = np.sqrt(((df['sinpkt'] - mean_ipt)**2 + (df['dinpkt'] - mean_ipt)**2) / 2)
        df['cv_ipt'] = std_ipt / (mean_ipt + epsilon)
        features_created += 1
    
    print(f"[OK] Created {features_created} advanced global features")
    
    total_new_features = (len(src_dst_pairs) * 8) + features_created
    print(f"\n{'='*80}")
    print(f"FEATURE ENGINEERING COMPLETE")
    print(f"{'='*80}")
    print(f"Total new features created: {total_new_features}")
    print(f"Note: Use select_top_features_by_importance() to reduce to top 30 features")
    print(f"{'='*80}\n")
    
    return df


def _apply_log_transforms_unsw(df):
    """
    Apply log1p transformation to highly skewed UNSW-NB15 features.
    Applied AFTER derived features are created to avoid affecting joint features.
    
    Creates new log-transformed columns with 'log_' prefix and removes originals:
        - Byte features: log_sbytes, log_dbytes
        - Packet features: log_spkts, log_dpkts
        - Duration: log_dur
        - Connection/flow statistics: log_ct_*, log_sloss, log_dloss, etc.
        - Derived features: log_sum_*, signed log_diff_*
        
    Original columns are removed after transformation to eliminate multicollinearity.
    """
    print(f"Applying log1p transformations to skewed features...")
    
    # Define base features that need log transformation
    skewed_base_features = [
        'sbytes', 'dbytes',      # Byte counts
        'spkts', 'dpkts',        # Packet counts
        'dur',                    # Duration
        'sloss', 'dloss',        # Loss counts
        'sttl', 'dttl',          # Time to live
        'sload', 'dload',        # Load
        'sjit', 'djit',          # Jitter (if exists)
    ]
    
    # Track transformations
    transformed_features = []
    features_to_remove = []
    
    # 1. Transform base features
    for base_feat in skewed_base_features:
        if base_feat in df.columns:
            # Create new log-transformed column
            log_col = f'log_{base_feat}'
            df[log_col] = np.log1p(df[base_feat])
            transformed_features.append(log_col)
            features_to_remove.append(base_feat)
    
    # 2. Transform derived sum features
    for col in df.columns:
        if col.startswith('sum_'):
            # Check if it's derived from skewed features
            if any(feat in col for feat in skewed_base_features):
                log_col = f'log_{col}'
                df[log_col] = np.log1p(df[col])
                transformed_features.append(log_col)
                features_to_remove.append(col)
    
    # 3. Transform derived diff features (preserve sign)
    for col in df.columns:
        if col.startswith('diff_'):
            if any(feat in col for feat in skewed_base_features):
                log_col = f'log_{col}'
                df[log_col] = np.sign(df[col]) * np.log1p(np.abs(df[col]))
                transformed_features.append(log_col)
                features_to_remove.append(col)
    
    # 4. Transform connection/flow count features
    ct_features = [col for col in df.columns if col.startswith('ct_')]
    for col in ct_features:
        log_col = f'log_{col}'
        df[log_col] = np.log1p(df[col])
        transformed_features.append(log_col)
        features_to_remove.append(col)
    
    # 5. Remove original features to avoid multicollinearity
    if features_to_remove:
        print(f"Created {len(transformed_features)} log-transformed features")
        print(f"Removing {len(features_to_remove)} original features to avoid multicollinearity")
        df.drop(columns=features_to_remove, inplace=True)
    
    return df


def _encode_unsw_categoricals(df):
    """
    DEPRECATED: Categorical encoding now handled per-model.
    
    This function previously converted categorical features (proto, service, 
    state, attack_cat) to integers using pd.factorize(). However, this approach:
    - Creates arbitrary numeric ordering (e.g., 'DoS'=0 < 'Exploit'=1)
    - Loses interpretability (attack_cat=3 vs 'Fuzzers')
    - Can cause inconsistent encodings between train/test splits
    
    NEW APPROACH:
    Keep categorical features as strings in preprocessing. Encode per-model:
    - Logistic Regression: OneHotEncoder or pd.get_dummies()
    - XGBoost/LightGBM: .astype('category') with enable_categorical=True
    - Random Forest: pd.get_dummies() or leave as-is (handles mixed types)
    
    This function is retained for backwards compatibility but should not be called.
    """
    import warnings
    warnings.warn(
        "_encode_unsw_categoricals() is deprecated. Keep categorical features "
        "as strings and encode per-model as needed.",
        DeprecationWarning,
        stacklevel=2
    )
    return df


def load_unsw():
    """Load and preprocess UNSW-NB15 training and test sets."""
    paths = _get_unsw_paths()
    train_df = pd.read_csv(paths["train"])
    test_df = pd.read_csv(paths["test"])

    for df in [train_df, test_df]:
        df = _clean_dataframe(df)
        # Note: Keep categorical features (proto, service, state, attack_cat) as strings
        # They will be encoded per-model as needed (OneHot, Label, or Category dtype)
        df = _feature_engineer_unsw(df)        # Create ALL derived features (basic + advanced)
        df = _apply_log_transforms_unsw(df)    # Then apply log transforms
        df = scale_numeric_features(df)        # Auto-detect and standardize

    print(f"UNSW-NB15 splits loaded: Train {train_df.shape}, Test {test_df.shape}")
    return train_df, test_df


# ================================================================
# ------------------ Combined Loader ------------------------------
# ================================================================

def prepare_datasets():
    """
    Load and preprocess both datasets from KaggleHub.
    Returns:
        beth_train, beth_val, beth_test, unsw_train, unsw_test
    """
    beth_train, beth_val, beth_test = load_beth()
    unsw_train, unsw_test = load_unsw()
    print("All datasets prepared successfully.")
    return beth_train, beth_val, beth_test, unsw_train, unsw_test


# ================================================================
# ------------------ Feature Selection ----------------------------
# ================================================================

def select_top_features_by_importance(X, y, n_features=30, method='random_forest', verbose=True):
    """
    Select top N features using model-based feature importance.
    
    This reduces dimensionality to a manageable set of most predictive features,
    improving model interpretability and reducing overfitting risk.
    
    Args:
        X: DataFrame of features (after preprocessing)
        y: Target variable
        n_features: Number of features to select (default: 30)
        method: Feature importance method ('random_forest', 'mutual_info', 'variance')
        verbose: Print selection details
        
    Returns:
        X_selected: DataFrame with top N features
        feature_importance_df: DataFrame with feature rankings
        selected_features: List of selected feature names
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Ensure we don't select more features than available
    n_features = min(n_features, X.shape[1])
    
    if verbose:
        print("\n" + "="*80)
        print(f"FEATURE SELECTION: Top {n_features} Features")
        print("="*80)
        print(f"Method: {method}")
        print(f"Original features: {X.shape[1]}")
    
    # Get only numeric features for importance calculation
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    if method == 'random_forest':
        # Use RandomForest feature importances
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(X_numeric, y)
        importances = rf.feature_importances_
        
        feature_importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
    elif method == 'mutual_info':
        # Use mutual information
        mi_scores = mutual_info_classif(X_numeric, y, random_state=42)
        
        feature_importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': mi_scores
        }).sort_values('importance', ascending=False)
        
    elif method == 'variance':
        # Use variance as proxy for importance
        variances = X_numeric.var()
        
        feature_importance_df = pd.DataFrame({
            'feature': numeric_cols,
            'importance': variances
        }).sort_values('importance', ascending=False)
        
    else:
        raise ValueError(f"Unknown method: {method}. Use 'random_forest', 'mutual_info', or 'variance'")
    
    # Select top N features
    selected_features = feature_importance_df.head(n_features)['feature'].tolist()
    
    # Include categorical features (if any remain as strings)
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_cols:
        selected_features.extend(categorical_cols)
        if verbose:
            print(f"Including {len(categorical_cols)} categorical features: {categorical_cols}")
    
    X_selected = X[selected_features]
    
    if verbose:
        print(f"Selected features: {len(selected_features)}")
        print(f"\nTop 15 features by importance:")
        print(feature_importance_df.head(15).to_string(index=False))
        print("="*80 + "\n")
    
    return X_selected, feature_importance_df, selected_features


def apply_feature_selection(df, target_col='label', n_features=30, method='random_forest', verbose=True):
    """
    Convenience function to apply feature selection to a preprocessed dataset.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column (default: 'label')
        n_features: Number of features to select (default: 30)
        method: Feature importance method
        verbose: Print selection details
        
    Returns:
        df_selected: DataFrame with selected features + target
        feature_importance: DataFrame with feature rankings
        selected_features: List of selected feature names
    """
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Select features
    X_selected, feature_importance, selected_features = select_top_features_by_importance(
        X, y, n_features=n_features, method=method, verbose=verbose
    )
    
    # Recombine with target
    df_selected = X_selected.copy()
    df_selected[target_col] = y
    
    return df_selected, feature_importance, selected_features


def cleanup_tfidf_text_columns(df, inplace=False):
    """
    Remove intermediate text columns created during TF-IDF feature engineering.
    
    This function removes the text columns that are created as intermediate steps
    during TF-IDF vectorization of the 'args' column but should not be kept in
    the final feature set.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame that may contain intermediate text columns
    inplace : bool, default=False
        If True, modify the DataFrame in place. If False, return a copy.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with text columns removed (or None if inplace=True)
    
    Example:
    --------
    >>> beth_train = cleanup_tfidf_text_columns(beth_train, inplace=True)
    >>> # Or for multiple DataFrames:
    >>> for df in [beth_train, beth_val, beth_test]:
    >>>     cleanup_tfidf_text_columns(df, inplace=True)
    """
    text_cols_to_remove = ['args', 'arg_names', 'arg_types', 'arg_values', 'args_str']
    existing_cols = [col for col in text_cols_to_remove if col in df.columns]
    
    if existing_cols:
        print(f"Removing {len(existing_cols)} TF-IDF text columns: {existing_cols}")
        if inplace:
            df.drop(columns=existing_cols, inplace=True)
            return None
        else:
            return df.drop(columns=existing_cols)
    else:
        print("No TF-IDF text columns found to remove")
        if inplace:
            return None
        else:
            return df.copy()