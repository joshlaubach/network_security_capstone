"""
Hyperparameter Tuning Utilities for Machine Learning Models

This module provides reusable functions for hyperparameter optimization with
progress tracking, time estimation, and comprehensive result reporting.

Author: Joshua Laubach
Date: October 28, 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from tqdm.auto import tqdm
from time import time

# Supervised learning
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Unsupervised learning
from sklearn.metrics import silhouette_score


class ProgressGridSearchCV(GridSearchCV):
    """
    GridSearchCV with better progress reporting.
    
    Note: sklearn's GridSearchCV batches candidates for efficiency,
    so tqdm can't show per-configuration progress. Instead, we use
    sklearn's built-in verbose parameter which prints progress.
    """
    
    def __init__(
        self,
        estimator,
        param_grid,
        *,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=2,  # Default to verbose=2 for progress
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        """Initialize with explicit parameters and verbose=2 by default."""
        super().__init__(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )


class ProgressRandomizedSearchCV(RandomizedSearchCV):
    """
    RandomizedSearchCV with better progress reporting.
    
    Note: sklearn's RandomizedSearchCV batches candidates for efficiency,
    so tqdm can't show per-configuration progress. Instead, we use
    sklearn's built-in verbose parameter which prints progress.
    """
    
    def __init__(
        self,
        estimator,
        param_distributions,
        *,
        n_iter=10,
        scoring=None,
        n_jobs=None,
        refit=True,
        cv=None,
        verbose=2,  # Default to verbose=2 for progress
        pre_dispatch="2*n_jobs",
        random_state=None,
        error_score=np.nan,
        return_train_score=False,
    ):
        """Initialize with explicit parameters and verbose=2 by default."""
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            random_state=random_state,
            error_score=error_score,
            return_train_score=return_train_score,
        )


# ============================================================================
# SUPERVISED LEARNING - HYPERPARAMETER TUNING
# ============================================================================

def tune_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    param_grid: Optional[Dict] = None,
    cv: int = 5,
    scoring: str = 'roc_auc',
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Any, Dict, float, pd.DataFrame]:
    """
    Perform hyperparameter tuning for Logistic Regression with progress tracking.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training labels
    X_val : pd.DataFrame, optional
        Validation features for final evaluation
    y_val : pd.Series, optional
        Validation labels for final evaluation
    param_grid : dict, optional
        Parameter grid. If None, uses default grid.
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric for optimization
    n_jobs : int
        Number of parallel jobs
    verbose : bool
        Whether to print progress information
    
    Returns:
    --------
    best_estimator : Pipeline
        Best fitted pipeline
    best_params : dict
        Best hyperparameters found
    best_score : float
        Best cross-validation score
    results_df : pd.DataFrame
        DataFrame with all results sorted by score
    """
    if verbose:
        print("="*80)
        print("HYPERPARAMETER TUNING - LOGISTIC REGRESSION")
        print("="*80)
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000]
        }
    
    # Create pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Prefix parameter names for pipeline
    param_grid_pipeline = {
        f'classifier__{key}': value for key, value in param_grid.items()
    }
    
    # Setup GridSearchCV with progress reporting
    start_time = time()
    
    grid_search = ProgressGridSearchCV(
        pipeline,
        param_grid=param_grid_pipeline,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        # verbose=2 is set by ProgressGridSearchCV for progress updates
        return_train_score=True
    )
    
    if verbose:
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nSearching {total_combinations} parameter combinations...")
        print(f"Cross-validation folds: {cv}")
        print(f"Scoring metric: {scoring}")
        print(f"\nProgress (sklearn verbose output):")
        print("-" * 60)
    
    # Fit with progress tracking
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time() - start_time
    
    # Get results
    best_params = {k.replace('classifier__', ''): v 
                   for k, v in grid_search.best_params_.items()}
    best_score = grid_search.best_score_
    
    # Create results DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ].head(10)
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed_time:.1f} seconds")
        print(f"\n[Best Parameters]")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\n[Best CV {scoring.upper()} Score]: {best_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = roc_auc_score(y_val, grid_search.predict_proba(X_val)[:, 1])
            print(f"[Validation {scoring.upper()} Score]: {val_score:.4f}")
        
        print("="*80)
    
    return grid_search.best_estimator_, best_params, best_score, results_df


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    param_grid: Optional[Dict] = None,
    cv: int = 3,
    scoring: str = 'roc_auc',
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Any, Dict, float, pd.DataFrame]:
    """
    Perform hyperparameter tuning for Random Forest with progress tracking.
    
    Parameters: Same as tune_logistic_regression
    
    Returns: Same as tune_logistic_regression
    """
    if verbose:
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING - RANDOM FOREST")
        print("="*80)
    
    # Default parameter grid
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2']
        }
    
    # Create base model
    rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Setup GridSearchCV with progress reporting
    start_time = time()
    
    grid_search = ProgressGridSearchCV(
        rf_base,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        # verbose=2 is set by ProgressGridSearchCV for progress updates
        return_train_score=True
    )
    
    if verbose:
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"\nSearching {total_combinations} parameter combinations...")
        print(f"Cross-validation folds: {cv}")
        print(f"Scoring metric: {scoring}")
        print(f"\nProgress (sklearn verbose output):")
        print("-" * 60)
    
    # Fit with progress tracking
    grid_search.fit(X_train, y_train)
    
    elapsed_time = time() - start_time
    
    # Get results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Create results DataFrame
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ].head(10)
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed_time:.1f} seconds")
        print(f"\n[Best Parameters]")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\n[Best CV {scoring.upper()} Score]: {best_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = roc_auc_score(y_val, grid_search.predict_proba(X_val)[:, 1])
            print(f"[Validation {scoring.upper()} Score]: {val_score:.4f}")
        
        print("="*80)
    
    return grid_search.best_estimator_, best_params, best_score, results_df


def tune_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    param_distributions: Optional[Dict] = None,
    n_iter: int = 50,
    cv: int = 3,
    scoring: str = 'roc_auc',
    n_jobs: int = -1,
    verbose: bool = True
) -> Tuple[Any, Dict, float, pd.DataFrame]:
    """
    Perform hyperparameter tuning for XGBoost with randomized search and progress tracking.
    
    Parameters:
    -----------
    param_distributions : dict, optional
        Parameter distributions for randomized search
    n_iter : int
        Number of random parameter combinations to try
    Other parameters: Same as tune_logistic_regression
    
    Returns: Same as tune_logistic_regression
    """
    if verbose:
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING - XGBOOST")
        print("="*80)
    
    # Default parameter distributions
    if param_distributions is None:
        param_distributions = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'reg_alpha': [0, 0.1, 1],
            'reg_lambda': [1, 2, 5]
        }
    
    # Create base model
    xgb_base = xgb.XGBClassifier(
        random_state=42,
        n_jobs=-1,
        objective='binary:logistic',
        eval_metric='logloss'
    )
    
    # Setup RandomizedSearchCV with progress reporting
    start_time = time()
    
    random_search = ProgressRandomizedSearchCV(
        xgb_base,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        # verbose=2 is set by ProgressRandomizedSearchCV for progress updates
        random_state=42,
        return_train_score=True
    )
    
    if verbose:
        print(f"\nTesting {n_iter} random parameter combinations...")
        print(f"Cross-validation folds: {cv}")
        print(f"Scoring metric: {scoring}")
        print(f"\nProgress (sklearn verbose output):")
        print("-" * 60)
    
    # Fit with progress tracking
    random_search.fit(X_train, y_train)
    
    elapsed_time = time() - start_time
    
    # Get results
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    # Create results DataFrame
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')[
        ['params', 'mean_test_score', 'std_test_score', 'rank_test_score']
    ].head(10)
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed_time:.1f} seconds")
        print(f"\n[Best Parameters]")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\n[Best CV {scoring.upper()} Score]: {best_score:.4f}")
        
        if X_val is not None and y_val is not None:
            val_score = roc_auc_score(y_val, random_search.predict_proba(X_val)[:, 1])
            print(f"[Validation {scoring.upper()} Score]: {val_score:.4f}")
        
        print("="*80)
    
    return random_search.best_estimator_, best_params, best_score, results_df


# ============================================================================
# UNSUPERVISED LEARNING - HYPERPARAMETER TUNING
# ============================================================================

def tune_kmeans(
    X_train: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    n_clusters_range: Optional[List[int]] = None,
    contamination_range: Optional[List[float]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform hyperparameter tuning for K-Means clustering with progress tracking.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_true : pd.Series, optional
        True labels for F1-score calculation (if available)
    n_clusters_range : list, optional
        Range of cluster numbers to try
    contamination_range : list, optional
        Range of contamination values to try
    random_state : int
        Random state for reproducibility
    verbose : bool
        Whether to print progress
    
    Returns:
    --------
    best_params : dict
        Best hyperparameters found
    results_df : pd.DataFrame
        DataFrame with all results sorted by score
    """
    from models_unsupervised import KMeansAnomalyDetector
    
    if verbose:
        print("="*80)
        print("HYPERPARAMETER TUNING - K-MEANS")
        print("="*80)
    
    # Default parameter ranges
    if n_clusters_range is None:
        n_clusters_range = [5, 6, 7, 8, 9, 10]
    if contamination_range is None:
        contamination_range = [0.03, 0.05, 0.07, 0.10]
    
    total_combinations = len(n_clusters_range) * len(contamination_range)
    
    if verbose:
        print(f"\nTesting {total_combinations} parameter combinations...\n")
    
    results = []
    best_score = -1
    best_params = {}
    
    start_time = time()
    
    with tqdm(total=total_combinations, desc="K-Means Grid Search", unit="config") as pbar:
        for n_clusters in n_clusters_range:
            for contamination in contamination_range:
                # Train model
                model = KMeansAnomalyDetector(
                    n_clusters=n_clusters,
                    contamination=contamination,
                    random_state=random_state
                )
                model.fit(X_train)
                
                # Calculate metrics
                labels = model.predict(X_train)
                cluster_labels = model.model.labels_
                sil_score = silhouette_score(X_train, cluster_labels)
                
                result = {
                    'n_clusters': n_clusters,
                    'contamination': contamination,
                    'silhouette_score': sil_score
                }
                
                # Add F1-score if labels available
                if y_true is not None:
                    f1 = f1_score(y_true, labels)
                    result['f1_score'] = f1
                
                results.append(result)
                
                # Update best parameters
                if sil_score > best_score:
                    best_score = sil_score
                    best_params = {
                        'n_clusters': n_clusters,
                        'contamination': contamination
                    }
                
                pbar.update(1)
    
    elapsed_time = time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('silhouette_score', ascending=False)
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed_time:.1f} seconds")
        print(f"\n[Best Parameters]")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\n[Best Silhouette Score]: {best_score:.4f}")
        print("\n[Top 5 Configurations]")
        print(results_df.head(5).to_string(index=False))
        print("="*80)
    
    return best_params, results_df


def tune_dbscan(
    X_train: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    eps_range: Optional[List[float]] = None,
    min_samples_range: Optional[List[int]] = None,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform hyperparameter tuning for DBSCAN with progress tracking.
    
    Parameters: Similar to tune_kmeans
    
    Returns: Same as tune_kmeans
    """
    from models_unsupervised import DBSCANAnomalyDetector
    
    if verbose:
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING - DBSCAN")
        print("="*80)
    
    # Default parameter ranges
    if eps_range is None:
        eps_range = [0.3, 0.5, 0.7, 1.0]
    if min_samples_range is None:
        min_samples_range = [3, 5, 7, 10]
    
    total_combinations = len(eps_range) * len(min_samples_range)
    
    if verbose:
        print(f"\nTesting {total_combinations} parameter combinations...\n")
    
    results = []
    best_score = -1
    best_params = {}
    
    start_time = time()
    
    with tqdm(total=total_combinations, desc="DBSCAN Grid Search", unit="config") as pbar:
        for eps in eps_range:
            for min_samples in min_samples_range:
                # Train model
                model = DBSCANAnomalyDetector(eps=eps, min_samples=min_samples)
                model.fit(X_train)
                
                # Get predictions
                labels = model.predict(X_train)
                cluster_labels = model.model.labels_
                
                # Calculate metrics
                n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
                
                # Silhouette score only for clustered points
                if n_clusters >= 2:
                    mask = cluster_labels != -1
                    if mask.sum() > 0:
                        sil_score = silhouette_score(X_train[mask], cluster_labels[mask])
                    else:
                        sil_score = -1
                else:
                    sil_score = -1
                
                outlier_rate = (labels == 1).sum() / len(labels)
                
                result = {
                    'eps': eps,
                    'min_samples': min_samples,
                    'silhouette_score': sil_score,
                    'n_clusters': n_clusters,
                    'outlier_rate': outlier_rate
                }
                
                # Add F1-score if labels available
                if y_true is not None:
                    f1 = f1_score(y_true, labels)
                    result['f1_score'] = f1
                    
                    # Use F1 as primary metric for DBSCAN
                    if f1 > best_score:
                        best_score = f1
                        best_params = {'eps': eps, 'min_samples': min_samples}
                
                results.append(result)
                pbar.update(1)
    
    elapsed_time = time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    if y_true is not None:
        results_df = results_df.sort_values('f1_score', ascending=False)
    else:
        results_df = results_df.sort_values('silhouette_score', ascending=False)
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed_time:.1f} seconds")
        print(f"\n[Best Parameters]")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        if y_true is not None:
            print(f"\n[Best F1-Score]: {best_score:.4f}")
        print("\n[Top 5 Configurations]")
        print(results_df.head(5).to_string(index=False))
        print("="*80)
    
    return best_params, results_df


def tune_gmm(
    X_train: pd.DataFrame,
    y_true: Optional[pd.Series] = None,
    n_components_range: Optional[List[int]] = None,
    covariance_types: Optional[List[str]] = None,
    contamination_range: Optional[List[float]] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[Dict, pd.DataFrame]:
    """
    Perform hyperparameter tuning for GMM with progress tracking.
    
    Parameters: Similar to tune_kmeans with additional covariance_types
    
    Returns: Same as tune_kmeans
    """
    from models_unsupervised import GMManomalyDetector
    
    if verbose:
        print("\n" + "="*80)
        print("HYPERPARAMETER TUNING - GMM")
        print("="*80)
    
    # Default parameter ranges
    if n_components_range is None:
        n_components_range = [3, 5, 7, 9, 10]
    if covariance_types is None:
        covariance_types = ['full', 'tied', 'diag', 'spherical']
    if contamination_range is None:
        contamination_range = [0.03, 0.05, 0.07]
    
    total_combinations = (len(n_components_range) * len(covariance_types) * 
                         len(contamination_range))
    
    if verbose:
        print(f"\nTesting {total_combinations} parameter combinations...")
        print("Using BIC (Bayesian Information Criterion) - lower is better\n")
    
    results = []
    best_bic = float('inf')
    best_params = {}
    
    start_time = time()
    
    with tqdm(total=total_combinations, desc="GMM Grid Search", unit="config") as pbar:
        for n_components in n_components_range:
            for cov_type in covariance_types:
                for contamination in contamination_range:
                    try:
                        # Train model
                        model = GMManomalyDetector(
                            n_components=n_components,
                            covariance_type=cov_type,
                            contamination=contamination,
                            random_state=random_state
                        )
                        model.fit(X_train)
                        
                        # Get predictions
                        labels = model.predict(X_train)
                        
                        # Calculate metrics
                        bic = model.model.bic(X_train)
                        aic = model.model.aic(X_train)
                        converged = model.model.converged_
                        
                        result = {
                            'n_components': n_components,
                            'covariance_type': cov_type,
                            'contamination': contamination,
                            'bic': bic,
                            'aic': aic,
                            'converged': converged
                        }
                        
                        # Add F1-score if labels available
                        if y_true is not None:
                            f1 = f1_score(y_true, labels)
                            result['f1_score'] = f1
                        
                        # Update best parameters (converged models only)
                        if converged and bic < best_bic:
                            best_bic = bic
                            best_params = {
                                'n_components': n_components,
                                'covariance_type': cov_type,
                                'contamination': contamination
                            }
                        
                        results.append(result)
                        
                    except Exception as e:
                        if verbose:
                            print(f"  Failed: n_comp={n_components}, cov={cov_type}, "
                                  f"contam={contamination:.2f} - {str(e)[:50]}")
                    
                    pbar.update(1)
    
    elapsed_time = time() - start_time
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    converged_df = results_df[results_df['converged'] == True]
    converged_df = converged_df.sort_values('bic')
    
    if verbose:
        print(f"\n[OK] Completed in {elapsed_time:.1f} seconds")
        print(f"\n[Best Parameters]")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        print(f"\n[Best BIC Score]: {best_bic:.2f} (lower is better)")
        print("\n[Top 5 Converged Configurations by BIC]")
        print(converged_df.head(5).to_string(index=False))
        print("="*80)
    
    return best_params, converged_df
