"""
feature_selection.py
--------------------
Feature selection methods for dimensionality reduction and model improvement.

Includes:
  - V    
    # Select features
    selected_features = [col for col in X_numeric.columns if col not in to_remove]
    X_selected = X_numeric[selected_features]
    
    if verbose:
        print(f"[Correlation-Based Selection]")
        print(f"  Threshold: {threshold}")
        print(f"  Original features: {X_numeric.shape[1]}")
        print(f"  Selected features: {X_selected.shape[1]}")
        print(f"  Removed features: {len(to_remove)}")
        print(f"  High correlation pairs found: {len(correlation_pairs)}")
    
    return X_selected, list(to_remove), correlation_pairsselection (VarianceThreshold)
  - Correlation-based selection (remove highly correlated features)
  - Univariate statistical tests (SelectKBest with chi2, f_classif, mutual_info)
  - Recursive Feature Elimination (RFE)
  - Model-based selection (SelectFromModel with tree-based models)
  - Combined feature selection pipeline
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    RFE,
    SelectFromModel,
    chi2,
    f_classif,
    mutual_info_classif
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# ----------------- Variance-Based Selection ---------------------
# ================================================================

def variance_threshold_selection(X, y=None, threshold=0.01, verbose=True):
    """
    Remove features with variance below threshold.
    
    Automatically filters to numeric columns only for robustness.
    
    Args:
        X: DataFrame or array of features
        y: Target variable (not used, for API consistency)
        threshold: Minimum variance threshold (default: 0.01)
        verbose: Print selection info
        
    Returns:
        X_selected: DataFrame with selected features
        selector: Fitted VarianceThreshold object
        selected_features: List of selected feature names
    """
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    
    # Filter to numeric columns only for robustness
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X.columns):
            if verbose:
                print(f"[INFO] Filtering to {len(numeric_cols)} numeric columns (excluded {len(X.columns) - len(numeric_cols)} non-numeric)")
        X_numeric = X[numeric_cols]
    else:
        X_numeric = X
        numeric_cols = None
    
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X_numeric)
    
    if feature_names is not None and numeric_cols is not None:
        selected_mask = selector.get_support()
        selected_features = numeric_cols[selected_mask].tolist()
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    else:
        selected_features = list(range(X_selected.shape[1]))
    
    if verbose:
        n_removed = X_numeric.shape[1] - X_selected.shape[1]
        print(f"[Variance Threshold Selection]")
        print(f"  Threshold: {threshold}")
        print(f"  Original features: {X_numeric.shape[1]}")
        print(f"  Selected features: {X_selected.shape[1]}")
        print(f"  Removed features: {n_removed}")
    
    return X_selected, selector, selected_features


# ================================================================
# ----------------- Correlation-Based Selection ------------------
# ================================================================

def correlation_selection(X, y=None, threshold=0.95, verbose=True):
    """
    Remove highly correlated features.
    
    Automatically filters to numeric columns only for robustness.
    
    Args:
        X: DataFrame of features
        y: Target variable (not used, for API consistency)
        threshold: Correlation threshold (default: 0.95)
        verbose: Print selection info
        
    Returns:
        X_selected: DataFrame with selected features
        removed_features: List of removed feature names
        correlation_pairs: List of (feature1, feature2, correlation) tuples
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Filter to numeric columns only for robustness
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < len(X.columns):
        if verbose:
            print(f"[INFO] Filtering to {len(numeric_cols)} numeric columns (excluded {len(X.columns) - len(numeric_cols)} non-numeric)")
    X_numeric = X[numeric_cols]
    
    # Compute correlation matrix
    corr_matrix = X_numeric.corr().abs()
    
    # Find pairs above threshold
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features to remove
    to_remove = set()
    correlation_pairs = []
    
    for column in upper_tri.columns:
        high_corr = upper_tri[column][upper_tri[column] > threshold]
        for idx in high_corr.index:
            correlation_pairs.append((column, idx, upper_tri[column][idx]))
            # Remove the feature with higher mean correlation
            if corr_matrix[column].mean() > corr_matrix[idx].mean():
                to_remove.add(column)
            else:
                to_remove.add(idx)
    
    # Select features
    selected_features = [col for col in X.columns if col not in to_remove]
    X_selected = X[selected_features]
    
    if verbose:
        print(f"[Correlation-Based Selection]")
        print(f"  Threshold: {threshold}")
        print(f"  Original features: {X.shape[1]}")
        print(f"  Selected features: {len(selected_features)}")
        print(f"  Removed features: {len(to_remove)}")
        print(f"  High correlation pairs found: {len(correlation_pairs)}")
    
    return X_selected, list(to_remove), correlation_pairs


def plot_correlation_matrix(X, top_n=50, figsize=(12, 10)):
    """
    Plot correlation matrix for top N features by variance.
    
    Args:
        X: DataFrame of features
        top_n: Number of features to display
        figsize: Figure size
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    # Select top N features by variance
    if X.shape[1] > top_n:
        variances = X.var().sort_values(ascending=False)
        top_features = variances.head(top_n).index.tolist()
        X_subset = X[top_features]
    else:
        X_subset = X
    
    # Compute correlation
    corr_matrix = X_subset.corr()
    
    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title(f'Feature Correlation Matrix (Top {len(X_subset.columns)} Features)',
              fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.show()


# ================================================================
# ----------------- Statistical Selection ------------------------
# ================================================================

def univariate_selection(X, y, score_func='f_classif', k=20, verbose=True):
    """
    Select top K features using univariate statistical tests.
    
    Automatically filters to numeric columns only for robustness.
    
    Args:
        X: DataFrame or array of features
        y: Target variable
        score_func: Scoring function ('f_classif', 'chi2', 'mutual_info')
        k: Number of features to select
        verbose: Print selection info
        
    Returns:
        X_selected: DataFrame with selected features
        selector: Fitted SelectKBest object
        feature_scores: DataFrame with feature names and scores
    """
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    
    # Filter to numeric columns only for robustness
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X.columns):
            if verbose:
                print(f"[INFO] Filtering to {len(numeric_cols)} numeric columns (excluded {len(X.columns) - len(numeric_cols)} non-numeric)")
        X_numeric = X[numeric_cols]
        feature_names = numeric_cols
    else:
        X_numeric = X
    
    # Choose score function
    score_funcs = {
        'f_classif': f_classif,
        'chi2': chi2,
        'mutual_info': mutual_info_classif
    }
    
    if score_func not in score_funcs:
        raise ValueError(f"score_func must be one of {list(score_funcs.keys())}")
    
    # For chi2, ensure non-negative values
    if score_func == 'chi2':
        X_numeric = X_numeric - X_numeric.min() if hasattr(X_numeric, 'min') else X_numeric
    
    selector = SelectKBest(score_func=score_funcs[score_func], k=min(k, X_numeric.shape[1]))
    X_selected = selector.fit_transform(X_numeric, y)
    
    # Get feature scores
    if feature_names is not None:
        selected_mask = selector.get_support()
        selected_features = feature_names[selected_mask].tolist()
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        feature_scores = pd.DataFrame({
            'feature': feature_names,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
    else:
        selected_features = list(range(X_selected.shape[1]))
        feature_scores = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(len(selector.scores_))],
            'score': selector.scores_
        }).sort_values('score', ascending=False)
    
    if verbose:
        print(f"[Univariate Feature Selection - {score_func}]")
        print(f"  Original features: {X_numeric.shape[1]}")
        print(f"  Selected features: {X_selected.shape[1]}")
        print(f"\n  Top 10 features by score:")
        print(feature_scores.head(10).to_string(index=False))
    
    return X_selected, selector, feature_scores


# ================================================================
# ----------------- Recursive Feature Elimination ----------------
# ================================================================

def rfe_selection(X, y, estimator=None, n_features_to_select=20, step=1, verbose=True):
    """
    Recursive Feature Elimination.
    
    Automatically filters to numeric columns only for robustness.
    
    Args:
        X: DataFrame or array of features
        y: Target variable
        estimator: Model to use for feature ranking (default: RandomForest)
        n_features_to_select: Number of features to select
        step: Number of features to remove at each iteration
        verbose: Print selection info
        
    Returns:
        X_selected: DataFrame with selected features
        selector: Fitted RFE object
        feature_ranking: DataFrame with feature names and rankings
    """
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    
    # Filter to numeric columns only for robustness
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X.columns):
            if verbose:
                print(f"[INFO] Filtering to {len(numeric_cols)} numeric columns (excluded {len(X.columns) - len(numeric_cols)} non-numeric)")
        X_numeric = X[numeric_cols]
        feature_names = numeric_cols
    else:
        X_numeric = X
    
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    selector = RFE(
        estimator=estimator,
        n_features_to_select=min(n_features_to_select, X_numeric.shape[1]),
        step=step,
        verbose=0
    )
    
    X_selected = selector.fit_transform(X_numeric, y)
    
    if feature_names is not None:
        selected_mask = selector.get_support()
        selected_features = feature_names[selected_mask].tolist()
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        feature_ranking = pd.DataFrame({
            'feature': feature_names,
            'ranking': selector.ranking_,
            'selected': selected_mask
        }).sort_values('ranking')
    else:
        selected_features = list(range(X_selected.shape[1]))
        feature_ranking = pd.DataFrame({
            'feature': [f'feature_{i}' for i in range(X_numeric.shape[1])],
            'ranking': selector.ranking_,
            'selected': selector.get_support()
        }).sort_values('ranking')
    
    if verbose:
        print(f"[Recursive Feature Elimination]")
        print(f"  Estimator: {estimator.__class__.__name__}")
        print(f"  Original features: {X_numeric.shape[1]}")
        print(f"  Selected features: {X_selected.shape[1]}")
        print(f"\n  Top 10 selected features:")
        top_features = feature_ranking[feature_ranking['selected']].head(10)
        print(top_features.to_string(index=False))
    
    return X_selected, selector, feature_ranking


# ================================================================
# ----------------- Model-Based Selection ------------------------
# ================================================================

def model_based_selection(X, y, estimator=None, threshold='median', verbose=True):
    """
    Select features based on model feature importances.
    
    Automatically filters to numeric columns only for robustness.
    
    Args:
        X: DataFrame or array of features
        y: Target variable
        estimator: Model to use (default: RandomForest)
        threshold: Importance threshold ('mean', 'median', or float)
        verbose: Print selection info
        
    Returns:
        X_selected: DataFrame with selected features
        selector: Fitted SelectFromModel object
        feature_importances: DataFrame with feature names and importances
    """
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    
    # Filter to numeric columns only for robustness
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X.columns):
            if verbose:
                print(f"[INFO] Filtering to {len(numeric_cols)} numeric columns (excluded {len(X.columns) - len(numeric_cols)} non-numeric)")
        X_numeric = X[numeric_cols]
        feature_names = numeric_cols
    else:
        X_numeric = X
    
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    selector = SelectFromModel(estimator=estimator, threshold=threshold, prefit=False)
    X_selected = selector.fit_transform(X_numeric, y)
    
    # Get feature importances
    estimator_fitted = selector.estimator_
    
    if feature_names is not None:
        selected_mask = selector.get_support()
        selected_features = feature_names[selected_mask].tolist()
        X_selected = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        if hasattr(estimator_fitted, 'feature_importances_'):
            importances = estimator_fitted.feature_importances_
        elif hasattr(estimator_fitted, 'coef_'):
            importances = np.abs(estimator_fitted.coef_).flatten()
        else:
            importances = np.ones(len(feature_names))
        
        feature_importances = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'selected': selected_mask
        }).sort_values('importance', ascending=False)
    else:
        selected_features = list(range(X_selected.shape[1]))
        feature_importances = None
    
    if verbose:
        print(f"[Model-Based Feature Selection]")
        print(f"  Estimator: {estimator.__class__.__name__}")
        print(f"  Threshold: {threshold}")
        print(f"  Original features: {X_numeric.shape[1]}")
        print(f"  Selected features: {X_selected.shape[1]}")
        
        if feature_importances is not None:
            print(f"\n  Top 10 selected features by importance:")
            top_features = feature_importances[feature_importances['selected']].head(10)
            print(top_features.to_string(index=False))
    
    return X_selected, selector, feature_importances


# ================================================================
# ----------------- Lasso-Based Selection ------------------------
# ================================================================

def lasso_selection(X, y, verbose=True):
    """
    Select features using Lasso (L1 regularization).
    
    Automatically filters to numeric columns only for robustness.
    
    Args:
        X: DataFrame or array of features
        y: Target variable
        verbose: Print selection info
        
    Returns:
        X_selected: DataFrame with selected features
        selector: Fitted LassoCV object
        feature_coefficients: DataFrame with feature names and coefficients
    """
    feature_names = X.columns if isinstance(X, pd.DataFrame) else None
    
    # Filter to numeric columns only for robustness
    if isinstance(X, pd.DataFrame):
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < len(X.columns):
            if verbose:
                print(f"[INFO] Filtering to {len(numeric_cols)} numeric columns (excluded {len(X.columns) - len(numeric_cols)} non-numeric)")
        X_numeric = X[numeric_cols]
        feature_names = numeric_cols
    else:
        X_numeric = X
    
    # Fit LassoCV
    lasso = LassoCV(cv=5, random_state=42, n_jobs=-1, max_iter=10000)
    lasso.fit(X_numeric, y)
    
    # Select features with non-zero coefficients
    selected_mask = lasso.coef_ != 0
    X_selected = X_numeric.iloc[:, selected_mask] if isinstance(X_numeric, pd.DataFrame) else X_numeric[:, selected_mask]
    
    if feature_names is not None:
        selected_features = feature_names[selected_mask].tolist()
        
        feature_coefficients = pd.DataFrame({
            'feature': feature_names,
            'coefficient': lasso.coef_,
            'abs_coefficient': np.abs(lasso.coef_),
            'selected': selected_mask
        }).sort_values('abs_coefficient', ascending=False)
    else:
        selected_features = list(range(X_selected.shape[1]))
        feature_coefficients = None
    
    if verbose:
        print(f"[Lasso Feature Selection]")
        print(f"  Alpha (selected by CV): {lasso.alpha_:.6f}")
        print(f"  Original features: {X_numeric.shape[1]}")
        print(f"  Selected features: {X_selected.shape[1]}")
        print(f"  Features with zero coefficients: {(~selected_mask).sum()}")
        
        if feature_coefficients is not None:
            print(f"\n  Top 10 features by coefficient magnitude:")
            top_features = feature_coefficients[feature_coefficients['selected']].head(10)
            print(top_features[['feature', 'coefficient', 'abs_coefficient']].to_string(index=False))
    
    return X_selected, lasso, feature_coefficients


# ================================================================
# ----------------- Sequential Feature Selection ----------------
# ================================================================

def forward_selection(X, y, estimator=None, n_features_to_select=20, cv=3, 
                      scoring='accuracy', verbose=True):
    """
    Sequential Forward Feature Selection using cross-validation.
    
    Starts with an empty set and iteratively adds the feature that most 
    improves the model performance.
    
    Args:
        X: Feature matrix
        y: Target variable
        estimator: Sklearn estimator (default: RandomForestClassifier)
        n_features_to_select: Number of features to select
        cv: Number of cross-validation folds
        scoring: Scoring metric for cross-validation
        verbose: Print progress
    
    Returns:
        X_selected: Filtered feature matrix
        selector: Fitted SequentialFeatureSelector object
        ranking_df: DataFrame with feature selection order
    """
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.ensemble import RandomForestClassifier
    
    if verbose:
        print(f"\n[Sequential Forward Selection]")
        print(f"  Starting features: {X.shape[1]}")
        print(f"  Target features: {n_features_to_select}")
        print(f"  Cross-validation: {cv}-fold")
    
    # Filter to numeric features only
    X_numeric = X.select_dtypes(include=[np.number])
    if len(X_numeric.columns) < len(X.columns):
        dropped = len(X.columns) - len(X_numeric.columns)
        if verbose:
            print(f"  ⚠️  Dropped {dropped} non-numeric features")
    
    # Default estimator
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Sequential forward selection
    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_to_select,
        direction='forward',
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    
    if verbose:
        print(f"\n  Running forward selection (this may take a while)...")
    
    selector.fit(X_numeric, y)
    
    # Get selected features
    selected_mask = selector.get_support()
    selected_features = X_numeric.columns[selected_mask].tolist()
    
    X_selected = X_numeric[selected_features]
    
    # Create ranking DataFrame
    ranking_df = pd.DataFrame({
        'feature': X_numeric.columns,
        'selected': selected_mask,
        'rank': [i+1 if selected_mask[i] else X_numeric.shape[1] 
                 for i in range(len(selected_mask))]
    }).sort_values('rank')
    
    if verbose:
        print(f"\n  ✓ Forward selection complete!")
        print(f"  Features selected: {len(selected_features)}")
    
    return X_selected, selector, ranking_df


def backward_selection(X, y, estimator=None, n_features_to_select=20, cv=3,
                       scoring='accuracy', verbose=True):
    """
    Sequential Backward Feature Selection using cross-validation.
    
    Starts with all features and iteratively removes the feature that least
    impacts model performance.
    
    Args:
        X: Feature matrix
        y: Target variable
        estimator: Sklearn estimator (default: RandomForestClassifier)
        n_features_to_select: Number of features to select
        cv: Number of cross-validation folds
        scoring: Scoring metric for cross-validation
        verbose: Print progress
    
    Returns:
        X_selected: Filtered feature matrix
        selector: Fitted SequentialFeatureSelector object
        ranking_df: DataFrame with feature selection order
    """
    from sklearn.feature_selection import SequentialFeatureSelector
    from sklearn.ensemble import RandomForestClassifier
    
    if verbose:
        print(f"\n[Sequential Backward Selection]")
        print(f"  Starting features: {X.shape[1]}")
        print(f"  Target features: {n_features_to_select}")
        print(f"  Cross-validation: {cv}-fold")
    
    # Filter to numeric features only
    X_numeric = X.select_dtypes(include=[np.number])
    if len(X_numeric.columns) < len(X.columns):
        dropped = len(X.columns) - len(X_numeric.columns)
        if verbose:
            print(f"  ⚠️  Dropped {dropped} non-numeric features")
    
    # Default estimator
    if estimator is None:
        estimator = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    # Sequential backward selection
    selector = SequentialFeatureSelector(
        estimator,
        n_features_to_select=n_features_to_select,
        direction='backward',
        scoring=scoring,
        cv=cv,
        n_jobs=-1
    )
    
    if verbose:
        print(f"\n  Running backward selection (this may take a while)...")
    
    selector.fit(X_numeric, y)
    
    # Get selected features
    selected_mask = selector.get_support()
    selected_features = X_numeric.columns[selected_mask].tolist()
    
    X_selected = X_numeric[selected_features]
    
    # Create ranking DataFrame
    ranking_df = pd.DataFrame({
        'feature': X_numeric.columns,
        'selected': selected_mask,
        'rank': [i+1 if selected_mask[i] else X_numeric.shape[1] 
                 for i in range(len(selected_mask))]
    }).sort_values('rank')
    
    if verbose:
        print(f"\n  ✓ Backward selection complete!")
        print(f"  Features selected: {len(selected_features)}")
    
    return X_selected, selector, ranking_df


# ================================================================
# ----------------- Combined Selection Pipeline ------------------
# ================================================================

def combined_selection(X, y, methods=['variance', 'correlation', 'model_based'], 
                      variance_threshold=0.01, corr_threshold=0.95, 
                      n_features=20, verbose=True):
    """
    Apply multiple feature selection methods sequentially.
    
    Args:
        X: DataFrame of features
        y: Target variable
        methods: List of methods to apply in order
        variance_threshold: Threshold for variance selection
        corr_threshold: Threshold for correlation selection
        n_features: Number of features for final selection
        verbose: Print selection info
        
    Returns:
        X_selected: DataFrame with selected features
        selection_results: Dictionary with results from each method
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    
    X_current = X.copy()
    selection_results = {}
    
    if verbose:
        print("=" * 80)
        print("COMBINED FEATURE SELECTION PIPELINE")
        print("=" * 80)
        print(f"Initial features: {X.shape[1]}\n")
    
    # Apply each method
    for method in methods:
        if method == 'variance' and 'variance' not in selection_results:
            X_current, selector, features = variance_threshold_selection(
                X_current, threshold=variance_threshold, verbose=verbose
            )
            selection_results['variance'] = {
                'selector': selector,
                'features': features,
                'n_features': len(features)
            }
            
        elif method == 'correlation' and 'correlation' not in selection_results:
            X_current, removed, pairs = correlation_selection(
                X_current, threshold=corr_threshold, verbose=verbose
            )
            selection_results['correlation'] = {
                'removed': removed,
                'pairs': pairs,
                'n_features': X_current.shape[1]
            }
            
        elif method == 'univariate' and 'univariate' not in selection_results:
            X_current, selector, scores = univariate_selection(
                X_current, y, k=min(n_features, X_current.shape[1]), verbose=verbose
            )
            selection_results['univariate'] = {
                'selector': selector,
                'scores': scores,
                'n_features': X_current.shape[1]
            }
            
        elif method == 'rfe' and 'rfe' not in selection_results:
            X_current, selector, ranking = rfe_selection(
                X_current, y, n_features_to_select=min(n_features, X_current.shape[1]), 
                verbose=verbose
            )
            selection_results['rfe'] = {
                'selector': selector,
                'ranking': ranking,
                'n_features': X_current.shape[1]
            }
            
        elif method == 'model_based' and 'model_based' not in selection_results:
            X_current, selector, importances = model_based_selection(
                X_current, y, verbose=verbose
            )
            selection_results['model_based'] = {
                'selector': selector,
                'importances': importances,
                'n_features': X_current.shape[1]
            }
        
        if verbose:
            print("")
    
    if verbose:
        print("=" * 80)
        print(f"FINAL RESULT: {X_current.shape[1]} features selected")
        print("=" * 80)
    
    return X_current, selection_results


# ================================================================
# ------------- Comprehensive Feature Selection Pipeline ---------
# ================================================================

def comprehensive_feature_selection(X_train, y_train, n_features=30, 
                                    skip_backward=True, verbose=True):
    """
    Run comprehensive feature selection pipeline with multiple methods and consensus analysis.
    
    This function:
    1. Runs 5-6 different feature selection methods
    2. Performs consensus analysis to identify features selected by multiple methods
    3. Returns all results including visualizations and selected features
    
    Args:
        X_train: Training feature matrix
        y_train: Training target variable
        n_features: Target number of features to select (default: 30)
        skip_backward: Skip backward selection (too slow) (default: True)
        verbose: Print progress and results (default: True)
        
    Returns:
        dict: Comprehensive results containing:
            - 'method_results': Dict of results from each method
            - 'consensus_df': DataFrame with consensus analysis
            - 'top_features': List of top consensus features
            - 'X_train_*': Selected feature matrices for each method
    """
    import time
    from sklearn.linear_model import LogisticRegression
    
    if verbose:
        print("\n" + "="*80)
        print("COMPREHENSIVE FEATURE SELECTION PIPELINE")
        print("="*80)
        print(f"Starting features: {X_train.shape[1]}")
        print(f"Target features: {n_features}")
        print(f"Methods: 5" + (" (backward selection skipped)" if skip_backward else " + backward selection"))
        print("="*80 + "\n")
    
    start_time = time.time()
    results = {}
    
    # ===== 1. Variance Threshold =====
    if verbose:
        print("\n[1/5] Running Variance Threshold Selection...")
    X_train_var, var_selector, var_features = variance_threshold_selection(
        X_train, threshold=0.01, verbose=verbose
    )
    results['variance'] = {
        'X_selected': X_train_var,
        'selector': var_selector,
        'features': var_features,
        'n_features': len(var_features)
    }
    
    # ===== 2. Correlation-Based =====
    if verbose:
        print("\n[2/5] Running Correlation-Based Selection...")
    X_train_corr, removed_features, corr_pairs = correlation_selection(
        X_train, threshold=0.95, verbose=verbose
    )
    results['correlation'] = {
        'X_selected': X_train_corr,
        'removed': removed_features,
        'pairs': corr_pairs,
        'n_features': X_train_corr.shape[1]
    }
    
    # ===== 3. Univariate F-test =====
    if verbose:
        print("\n[3/5] Running Univariate F-test Selection...")
    X_train_uni, uni_selector, uni_scores = univariate_selection(
        X_train, y_train, k=n_features, verbose=verbose
    )
    results['univariate'] = {
        'X_selected': X_train_uni,
        'selector': uni_selector,
        'scores': uni_scores,
        'n_features': X_train_uni.shape[1]
    }
    
    # ===== 4. Model-Based (Random Forest) =====
    if verbose:
        print("\n[4/5] Running Model-Based Selection (Random Forest)...")
    X_train_model, model_selector, model_importances = model_based_selection(
        X_train, y_train, threshold='median', verbose=verbose
    )
    results['model_based'] = {
        'X_selected': X_train_model,
        'selector': model_selector,
        'importances': model_importances,
        'n_features': X_train_model.shape[1]
    }
    
    # ===== 5. Lasso (L1 Regularization) =====
    if verbose:
        print("\n[5/5] Running Lasso Selection (L1 Regularization)...")
    X_train_lasso, lasso_selector, lasso_coeffs = lasso_selection(
        X_train, y_train, verbose=verbose
    )
    results['lasso'] = {
        'X_selected': X_train_lasso,
        'selector': lasso_selector,
        'coefficients': lasso_coeffs,
        'n_features': X_train_lasso.shape[1]
    }
    
    # ===== 6. Backward Selection (Optional - SLOW) =====
    if not skip_backward:
        if verbose:
            print("\n[OPTIONAL] Running Backward Selection...")
            print("   ⚠️  Warning: This may take 20+ minutes with LogisticRegression")
        
        fast_estimator = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        X_train_backward, backward_selector, backward_ranking = backward_selection(
            X_train, y_train, estimator=fast_estimator, 
            n_features_to_select=n_features, cv=3, verbose=verbose
        )
        results['backward'] = {
            'X_selected': X_train_backward,
            'selector': backward_selector,
            'ranking': backward_ranking,
            'n_features': X_train_backward.shape[1]
        }
    else:
        # Use univariate as proxy for backward
        if verbose:
            print("\n[OPTIONAL] Backward Selection SKIPPED (using univariate as proxy)")
        results['backward'] = {
            'X_selected': X_train_uni.copy(),
            'selector': None,
            'ranking': pd.DataFrame({
                'feature': X_train.columns,
                'selected': [f in X_train_uni.columns for f in X_train.columns]
            }),
            'n_features': X_train_uni.shape[1]
        }
    
    # ===== Consensus Analysis =====
    if verbose:
        print("\n" + "="*80)
        print("CONSENSUS ANALYSIS")
        print("="*80)
    
    # Collect feature sets from each method
    variance_features = set(results['variance']['X_selected'].columns)
    correlation_features = set(results['correlation']['X_selected'].columns)
    univariate_features = set(results['univariate']['X_selected'].columns)
    backward_features = set(results['backward']['X_selected'].columns)
    model_features = set(results['model_based']['X_selected'].columns)
    lasso_features = set(results['lasso']['X_selected'].columns)
    
    # Count consensus
    all_features = (variance_features | correlation_features | univariate_features | 
                   backward_features | model_features | lasso_features)
    
    consensus_count = {}
    for feature in all_features:
        count = sum([
            feature in variance_features,
            feature in correlation_features,
            feature in univariate_features,
            feature in backward_features,
            feature in model_features,
            feature in lasso_features
        ])
        consensus_count[feature] = count
    
    # Create consensus DataFrame
    consensus_df = pd.DataFrame({
        'feature': list(consensus_count.keys()),
        'consensus_count': list(consensus_count.values()),
        'variance': [f in variance_features for f in consensus_count.keys()],
        'correlation': [f in correlation_features for f in consensus_count.keys()],
        'univariate': [f in univariate_features for f in consensus_count.keys()],
        'backward': [f in backward_features for f in consensus_count.keys()],
        'model_based': [f in model_features for f in consensus_count.keys()],
        'lasso': [f in lasso_features for f in consensus_count.keys()]
    }).sort_values('consensus_count', ascending=False).reset_index(drop=True)
    
    if verbose:
        print(f"\nTotal unique features: {len(all_features)}")
        print("\nConsensus breakdown:")
        for i in range(6, 0, -1):
            count = sum(consensus_df['consensus_count'] == i)
            if count > 0:
                print(f"  {i}/6 methods: {count} features")
        
        print(f"\nTop 20 features by consensus:")
        print(consensus_df.head(20)[['feature', 'consensus_count']].to_string(index=False))
    
    # Get top consensus features
    top_features = consensus_df[consensus_df['consensus_count'] >= 4]['feature'].tolist()
    
    # Summary
    elapsed = time.time() - start_time
    if verbose:
        print("\n" + "="*80)
        print(f"PIPELINE COMPLETE in {elapsed:.1f} seconds")
        print(f"Top consensus features (4+ methods): {len(top_features)}")
        print("="*80 + "\n")
    
    return {
        'method_results': results,
        'consensus_df': consensus_df,
        'top_features': top_features,
        'feature_sets': {
            'variance': variance_features,
            'correlation': correlation_features,
            'univariate': univariate_features,
            'backward': backward_features,
            'model_based': model_features,
            'lasso': lasso_features
        }
    }


def visualize_consensus(consensus_df, figsize=(16, 6)):
    """
    Visualize feature selection consensus results.
    
    Args:
        consensus_df: DataFrame from comprehensive_feature_selection()
        figsize: Figure size (default: (16, 6))
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Consensus count distribution
    axes[0].bar(consensus_df['consensus_count'].value_counts().sort_index().index,
                consensus_df['consensus_count'].value_counts().sort_index().values,
                color='steelblue', edgecolor='black', alpha=0.8)
    axes[0].set_xlabel('Number of Methods Agreeing', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Number of Features', fontsize=11, fontweight='bold')
    axes[0].set_title('Feature Selection Consensus Distribution', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Heatmap of top consensus features
    top_consensus = consensus_df.head(20)
    heatmap_data = top_consensus[['variance', 'correlation', 'univariate', 
                                   'backward', 'model_based', 'lasso']].astype(int).values
    im = axes[1].imshow(heatmap_data, cmap='YlGn', aspect='auto')
    axes[1].set_yticks(range(len(top_consensus)))
    axes[1].set_yticklabels(top_consensus['feature'], fontsize=9)
    axes[1].set_xticks(range(6))
    axes[1].set_xticklabels(['Var', 'Corr', 'Uni', 'Back', 'Model', 'Lasso'], fontsize=10)
    axes[1].set_title('Top 20 Features by Method Agreement', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
