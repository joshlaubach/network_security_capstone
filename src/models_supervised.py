"""
models_supervised.py
--------------------
Supervised classification models for the UNSW-NB15 dataset.

Implements:
  - Logistic Regression (with feature scaling)
  - Random Forest (with feature importance analysis)
  - XGBoost (with hyperparameter tuning: learning rate, n_estimators, max_depth, regularization)

Target Variable:
  - Binary classification (normal vs attack) or multi-class attack types
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# Categorical Encoding Utilities
# ================================================================

def encode_categoricals(X, encoding='onehot', encoder=None, categorical_cols=None):
    """
    Encode categorical features for model compatibility.
    
    Args:
        X: Features (DataFrame)
        encoding: Encoding type - 'onehot', 'label', or 'category' (for XGBoost)
        encoder: Pre-fitted encoder object (optional, for test data)
        categorical_cols: List of categorical column names (auto-detected if None)
        
    Returns:
        Tuple of (encoded_X, encoder_object, categorical_cols)
    """
    if not isinstance(X, pd.DataFrame):
        return X, None, []
    
    # Auto-detect categorical columns if not provided
    if categorical_cols is None:
        categorical_cols = X.select_dtypes(include='object').columns.tolist()
    
    # No categorical columns to encode
    if len(categorical_cols) == 0:
        return X, None, []
    
    X_encoded = X.copy()
    
    if encoding == 'onehot':
        # One-hot encoding for Logistic Regression
        if encoder is None:
            from sklearn.preprocessing import OneHotEncoder
            encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            
            # Fit and transform
            encoded_cat = encoder.fit_transform(X[categorical_cols])
            feature_names = encoder.get_feature_names_out(categorical_cols)
            
            # Create DataFrame with encoded features
            encoded_df = pd.DataFrame(encoded_cat, columns=feature_names, index=X.index)
            
            # Drop original categorical columns and add encoded ones
            X_encoded = X.drop(columns=categorical_cols)
            X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
        else:
            # Transform only
            encoded_cat = encoder.transform(X[categorical_cols])
            feature_names = encoder.get_feature_names_out(categorical_cols)
            encoded_df = pd.DataFrame(encoded_cat, columns=feature_names, index=X.index)
            
            X_encoded = X.drop(columns=categorical_cols)
            X_encoded = pd.concat([X_encoded, encoded_df], axis=1)
    
    elif encoding == 'label':
        # Label encoding (for Random Forest if needed)
        from sklearn.preprocessing import LabelEncoder
        
        if encoder is None:
            encoder = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X_encoded[col] = le.fit_transform(X[col].astype(str))
                encoder[col] = le
        else:
            for col in categorical_cols:
                X_encoded[col] = encoder[col].transform(X[col].astype(str))
    
    elif encoding == 'category':
        # Category dtype for XGBoost with enable_categorical=True
        for col in categorical_cols:
            X_encoded[col] = X[col].astype('category')
        encoder = None  # No encoder needed for category dtype
    
    else:
        raise ValueError(f"Unknown encoding method: {encoding}")
    
    return X_encoded, encoder, categorical_cols


# ================================================================
# Feature Scaling Utilities
# ================================================================

def scale_features(X, method='standard', scaler=None):
    """
    Scale features using various methods.
    
    Args:
        X: Features to scale (DataFrame or array)
        method: Scaling method - 'standard', 'minmax', 'robust', or 'none'
        scaler: Pre-fitted scaler object (optional, for test data)
        
    Returns:
        Tuple of (scaled_X, scaler_object)
    """
    if method == 'none':
        return X, None
    
    is_dataframe = isinstance(X, pd.DataFrame)
    if is_dataframe:
        columns = X.columns
        index = X.index
        # Only scale numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X_values = X[numeric_cols].values
    else:
        X_values = X
        columns = None
        index = None
        numeric_cols = None
    
    if scaler is None:
        # Fit new scaler
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        X_scaled = scaler.fit_transform(X_values)
    else:
        # Use pre-fitted scaler
        X_scaled = scaler.transform(X_values)
    
    if is_dataframe:
        # Reconstruct DataFrame with scaled numeric columns
        X_result = X.copy()
        X_result[numeric_cols] = X_scaled
        return X_result, scaler
    else:
        return X_scaled, scaler


# ================================================================
# Logistic Regression Classifier
# ================================================================

class LogisticRegressionClassifier:
    """
    Logistic Regression classifier with feature scaling.
    
    Features:
      - Multiple scaling options (standard, minmax, robust)
      - L1/L2 regularization
      - Probability predictions
      - Feature coefficient analysis
    """
    
    def __init__(self, penalty='l2', C=1.0, max_iter=1000, random_state=42,
                 scaler='standard', solver='lbfgs', categorical_encoding='onehot'):
        """
        Args:
            penalty: Regularization type ('l1', 'l2', 'elasticnet', 'none')
            C: Inverse of regularization strength (smaller = stronger)
            max_iter: Maximum iterations for solver
            random_state: Random seed
            scaler: Feature scaling method ('standard', 'minmax', 'robust', 'none')
            solver: Algorithm to use ('lbfgs', 'liblinear', 'saga')
            categorical_encoding: How to encode categoricals ('onehot' recommended)
        """
        self.penalty = penalty
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self.scaler_method = scaler
        self.solver = solver
        self.categorical_encoding = categorical_encoding
        self.model = None
        self.scaler_obj = None
        self.encoder_obj = None
        self.categorical_cols = []
        self.feature_names = None
        
    def fit(self, X, y):
        """
        Fit Logistic Regression model.
        
        Args:
            X: Training features (DataFrame with categorical columns as strings)
            y: Training labels
        """
        # Encode categorical features (OneHot for Logistic Regression)
        X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals(
            X, encoding=self.categorical_encoding
        )
        
        # Store feature names after encoding
        if isinstance(X_encoded, pd.DataFrame):
            self.feature_names = X_encoded.columns.tolist()
        
        # Apply feature scaling to numeric features only
        X_scaled, self.scaler_obj = scale_features(X_encoded, method=self.scaler_method)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Initialize and fit model
        self.model = LogisticRegression(
            penalty=self.penalty,
            C=self.C,
            max_iter=self.max_iter,
            random_state=self.random_state,
            solver=self.solver
        )
        
        self.model.fit(X_scaled, y)
        
        print(f"[INFO] Logistic Regression fitted")
        print(f"       Penalty: {self.penalty}, C: {self.C}")
        print(f"       Categorical encoding: {self.categorical_encoding} ({len(self.categorical_cols)} cols)")
        print(f"       Scaling: {self.scaler_method}")
        print(f"       Solver: {self.solver}")
        print(f"       Training accuracy: {self.model.score(X_scaled, y):.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features to predict (with categorical columns as strings)
            
        Returns:
            Predicted class labels
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals(
            X, encoding=self.categorical_encoding, 
            encoder=self.encoder_obj, 
            categorical_cols=self.categorical_cols
        )
        
        # Scale features
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict (with categorical columns as strings)
            
        Returns:
            Probability matrix
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals(
            X, encoding=self.categorical_encoding,
            encoder=self.encoder_obj,
            categorical_cols=self.categorical_cols
        )
        
        # Scale features
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        return self.model.predict_proba(X_scaled)
    
    def get_coefficients(self, top_n=20, plot=True):
        """
        Get and visualize feature coefficients.
        
        Args:
            top_n: Number of top features to display
            plot: Whether to plot coefficients
            
        Returns:
            DataFrame of feature coefficients
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting coefficients")
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.coef_[0]))]
        else:
            feature_names = self.feature_names
        
        # Get coefficients
        coef = self.model.coef_[0] if len(self.model.coef_.shape) > 1 else self.model.coef_
        
        coef_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coef,
            'abs_coefficient': np.abs(coef)
        }).sort_values('abs_coefficient', ascending=False)
        
        if plot:
            top_features = coef_df.head(top_n)
            
            plt.figure(figsize=(10, 8))
            colors = ['red' if x < 0 else 'green' for x in top_features['coefficient']]
            plt.barh(range(len(top_features)), top_features['coefficient'], color=colors)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Coefficient Value', fontsize=12)
            plt.title(f'Top {top_n} Feature Coefficients - Logistic Regression', 
                     fontsize=14, fontweight='bold')
            plt.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
            plt.tight_layout()
            plt.show()
        
        return coef_df


# ================================================================
# Random Forest Classifier
# ================================================================

class RandomForestClassifierModel:
    """
    Random Forest classifier with feature importance analysis.
    
    Features:
      - Feature importance extraction and visualization
      - Out-of-bag score estimation
      - Probability predictions
      - Configurable tree parameters
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', random_state=42,
                 n_jobs=-1, oob_score=True, categorical_encoding='onehot'):
        """
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None = unlimited)
            min_samples_split: Minimum samples required to split node
            min_samples_leaf: Minimum samples required in leaf node
            max_features: Number of features for best split ('sqrt', 'log2', None)
            random_state: Random seed
            n_jobs: Number of parallel jobs (-1 = use all cores)
            oob_score: Whether to use out-of-bag samples for validation
            categorical_encoding: How to encode categoricals ('onehot' or 'label')
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.oob_score = oob_score
        self.categorical_encoding = categorical_encoding
        self.model = None
        self.feature_names = None
        self.encoder_obj = None
        self.categorical_cols = []
        
    def fit(self, X, y):
        """
        Fit Random Forest model.
        
        Args:
            X: Training features (DataFrame with categorical columns as strings)
            y: Training labels
        """
        # Encode categorical features (OneHot or Label for Random Forest)
        X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals(
            X, encoding=self.categorical_encoding
        )
        
        # Store feature names after encoding
        if isinstance(X_encoded, pd.DataFrame):
            self.feature_names = X_encoded.columns.tolist()
            X_encoded = X_encoded.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Initialize and fit model
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            oob_score=self.oob_score
        )
        
        self.model.fit(X_encoded, y)
        
        print(f"[INFO] Random Forest fitted")
        print(f"       Trees: {self.n_estimators}, Max Depth: {self.max_depth}")
        print(f"       Categorical encoding: {self.categorical_encoding} ({len(self.categorical_cols)} cols)")
        print(f"       Training accuracy: {self.model.score(X_encoded, y):.4f}")
        if self.oob_score:
            print(f"       OOB Score: {self.model.oob_score_:.4f}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features to predict (with categorical columns as strings)
            
        Returns:
            Predicted class labels
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals(
            X, encoding=self.categorical_encoding,
            encoder=self.encoder_obj,
            categorical_cols=self.categorical_cols
        )
        
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
            
        return self.model.predict(X_encoded)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict (with categorical columns as strings)
            
        Returns:
            Probability matrix
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals(
            X, encoding=self.categorical_encoding,
            encoder=self.encoder_obj,
            categorical_cols=self.categorical_cols
        )
        
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
            
        return self.model.predict_proba(X_encoded)
    
    def get_feature_importances(self, top_n=20, plot=True):
        """
        Get and visualize feature importances.
        
        Args:
            top_n: Number of top features to display
            plot: Whether to plot importances
            
        Returns:
            DataFrame of feature importances
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importances")
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.model.feature_importances_))]
        else:
            feature_names = self.feature_names
        
        # Get feature importances
        importances = self.model.feature_importances_
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if plot:
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - Random Forest', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        return importance_df
    
    def get_tree_depths(self):
        """
        Get statistics about tree depths in the forest.
        
        Returns:
            Dictionary with depth statistics
        """
        depths = [tree.get_depth() for tree in self.model.estimators_]
        
        return {
            'mean_depth': np.mean(depths),
            'max_depth': np.max(depths),
            'min_depth': np.min(depths),
            'std_depth': np.std(depths)
        }


# ================================================================
# XGBoost Classifier
# ================================================================

class XGBoostClassifier:
    """
    XGBoost classifier with comprehensive hyperparameter tuning.
    
    Features:
      - Learning rate control
      - Number of estimators (boosting rounds)
      - Tree depth configuration
      - L1/L2 regularization
      - Feature importance analysis
      - Early stopping support
    """
    
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=6,
                 min_child_weight=1, gamma=0, subsample=1.0, colsample_bytree=1.0,
                 reg_alpha=0, reg_lambda=1, random_state=42, n_jobs=-1,
                 early_stopping_rounds=None, eval_metric='logloss',
                 categorical_encoding='category', enable_categorical=True):
        """
        Args:
            n_estimators: Number of boosting rounds
            learning_rate: Step size shrinkage (eta)
            max_depth: Maximum tree depth
            min_child_weight: Minimum sum of instance weight in child
            gamma: Minimum loss reduction for split
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns when constructing each tree
            reg_alpha: L1 regularization term (alpha)
            reg_lambda: L2 regularization term (lambda)
            random_state: Random seed
            n_jobs: Number of parallel threads (-1 = use all cores)
            early_stopping_rounds: Stop if no improvement for N rounds
            eval_metric: Evaluation metric ('logloss', 'error', 'auc')
            categorical_encoding: How to encode categoricals ('category' recommended for XGBoost)
            enable_categorical: Enable native categorical feature support
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.gamma = gamma
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.categorical_encoding = categorical_encoding
        self.enable_categorical = enable_categorical
        self.model = None
        self.feature_names = None
        self.encoder_obj = None
        self.categorical_cols = []
        self.evals_result = {}
        
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        """
        Fit XGBoost model.
        
        Args:
            X: Training features (DataFrame with categorical columns as strings)
            y: Training labels
            X_val: Validation features (optional, for early stopping)
            y_val: Validation labels (optional, for early stopping)
            verbose: Whether to print training progress
        """
        # Encode categorical features (category dtype for XGBoost native support)
        X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals(
            X, encoding=self.categorical_encoding
        )
        
        # Store feature names after encoding
        if isinstance(X_encoded, pd.DataFrame):
            self.feature_names = X_encoded.columns.tolist()
            X_values = X_encoded.values if self.categorical_encoding != 'category' else X_encoded
        else:
            X_values = X_encoded
            
        if isinstance(y, pd.Series):
            y = y.values
        
        # Prepare validation set if provided
        eval_set = []
        if X_val is not None and y_val is not None:
            X_val_encoded, _, _ = encode_categoricals(
                X_val, encoding=self.categorical_encoding,
                encoder=self.encoder_obj,
                categorical_cols=self.categorical_cols
            )
            if isinstance(X_val_encoded, pd.DataFrame):
                X_val_values = X_val_encoded.values if self.categorical_encoding != 'category' else X_val_encoded
            else:
                X_val_values = X_val_encoded
            if isinstance(y_val, pd.Series):
                y_val = y_val.values
            eval_set = [(X_val_values, y_val)]
        
        # Determine objective based on number of classes
        n_classes = len(np.unique(y))
        if n_classes == 2:
            objective = 'binary:logistic'
        else:
            objective = 'multi:softprob'
        
        # Initialize model with early_stopping_rounds in constructor (XGBoost >= 2.0)
        xgb_kwargs = {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'objective': objective,
            'eval_metric': self.eval_metric,
            'enable_categorical': self.enable_categorical
        }
        
        # Add early_stopping_rounds to constructor if provided and eval_set exists
        if self.early_stopping_rounds is not None and eval_set:
            xgb_kwargs['early_stopping_rounds'] = self.early_stopping_rounds
        
        self.model = xgb.XGBClassifier(**xgb_kwargs)
        
        # Fit model
        if eval_set:
            self.model.fit(
                X_values, y,
                eval_set=eval_set,
                verbose=verbose
            )
            self.evals_result = self.model.evals_result()
        else:
            self.model.fit(X_values, y, verbose=verbose)
        
        print(f"[INFO] XGBoost fitted")
        print(f"       Estimators: {self.n_estimators}, Learning Rate: {self.learning_rate}")
        print(f"       Max Depth: {self.max_depth}, Gamma: {self.gamma}")
        print(f"       Regularization - L1 (alpha): {self.reg_alpha}, L2 (lambda): {self.reg_lambda}")
        print(f"       Categorical encoding: {self.categorical_encoding} ({len(self.categorical_cols)} cols)")
        print(f"       Enable categorical: {self.enable_categorical}")
        print(f"       Training accuracy: {self.model.score(X_values, y):.4f}")
        if hasattr(self.model, 'best_iteration'):
            print(f"       Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features to predict (with categorical columns as strings)
            
        Returns:
            Predicted class labels
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals(
            X, encoding=self.categorical_encoding,
            encoder=self.encoder_obj,
            categorical_cols=self.categorical_cols
        )
        
        if isinstance(X_encoded, pd.DataFrame):
            X_values = X_encoded.values if self.categorical_encoding != 'category' else X_encoded
        else:
            X_values = X_encoded
            
        return self.model.predict(X_values)
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features to predict (with categorical columns as strings)
            
        Returns:
            Probability matrix
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals(
            X, encoding=self.categorical_encoding,
            encoder=self.encoder_obj,
            categorical_cols=self.categorical_cols
        )
        
        if isinstance(X_encoded, pd.DataFrame):
            X_values = X_encoded.values if self.categorical_encoding != 'category' else X_encoded
        else:
            X_values = X_encoded
            
        return self.model.predict_proba(X_values)
    
    def get_feature_importances(self, importance_type='gain', top_n=20, plot=True):
        """
        Get and visualize feature importances.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            top_n: Number of top features to display
            plot: Whether to plot importances
            
        Returns:
            DataFrame of feature importances
        """
        if self.model is None:
            raise ValueError("Model must be fitted before getting feature importances")
        
        # Get feature importances
        importance_dict = self.model.get_booster().get_score(importance_type=importance_type)
        
        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance_dict))]
        else:
            feature_names = self.feature_names
        
        # Map feature indices to names
        importances = []
        for i, name in enumerate(feature_names):
            key = f'f{i}'
            importances.append(importance_dict.get(key, 0))
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        if plot:
            top_features = importance_df.head(top_n)
            
            plt.figure(figsize=(10, 8))
            plt.barh(range(len(top_features)), top_features['importance'], color='darkorange')
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel(f'Importance ({importance_type})', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances - XGBoost ({importance_type})', 
                     fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
        
        return importance_df
    
    def plot_learning_curve(self):
        """
        Plot learning curves from training history.
        """
        if not self.evals_result:
            print("[WARN] No evaluation results available. Use validation set during training.")
            return
        
        results = list(self.evals_result.values())[0]
        metric_name = list(results.keys())[0]
        metric_values = results[metric_name]
        
        plt.figure(figsize=(10, 6))
        plt.plot(metric_values, linewidth=2, color='blue')
        plt.xlabel('Boosting Round', fontsize=12)
        plt.ylabel(metric_name, fontsize=12)
        plt.title(f'XGBoost Learning Curve - {metric_name}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        if hasattr(self.model, 'best_iteration'):
            plt.axvline(x=self.model.best_iteration, color='red', linestyle='--',
                       label=f'Best iteration: {self.model.best_iteration}')
            plt.legend()
        plt.tight_layout()
        plt.show()
    
    def get_params_summary(self):
        """
        Get summary of model hyperparameters.
        
        Returns:
            Dictionary of hyperparameters
        """
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_child_weight': self.min_child_weight,
            'gamma': self.gamma,
            'subsample': self.subsample,
            'colsample_bytree': self.colsample_bytree,
            'reg_alpha': self.reg_alpha,
            'reg_lambda': self.reg_lambda,
            'eval_metric': self.eval_metric
        }


# ================================================================
# Utility Functions
# ================================================================

def train_all_models(X_train, y_train, X_val=None, y_val=None, scaler='standard'):
    """
    Train all three supervised models.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional, for XGBoost early stopping)
        y_val: Validation labels (optional)
        scaler: Scaling method for Logistic Regression
        
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*60)
    print("Training All Supervised Models")
    print("="*60)
    
    models = {}
    
    # Logistic Regression
    print("\n[1/3] Training Logistic Regression...")
    lr = LogisticRegressionClassifier(
        penalty='l2',
        C=1.0,
        scaler=scaler,
        random_state=42
    )
    lr.fit(X_train, y_train)
    models['logistic_regression'] = lr
    
    # Random Forest
    print("\n[2/3] Training Random Forest...")
    rf = RandomForestClassifierModel(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        oob_score=True
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    
    # XGBoost
    print("\n[3/3] Training XGBoost...")
    xgb_model = XGBoostClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        reg_alpha=0,
        reg_lambda=1,
        random_state=42
    )
    xgb_model.fit(X_train, y_train, X_val=X_val, y_val=y_val, verbose=False)
    models['xgboost'] = xgb_model
    
    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60 + "\n")
    
    return models


def compare_models(models, X_test, y_test):
    """
    Compare performance of all models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test labels
        
    Returns:
        DataFrame with model comparison
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    results = []
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        
        # Handle multi-class vs binary
        average = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
        
        results.append({
            'model': name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average=average, zero_division=0),
            'recall': recall_score(y_test, y_pred, average=average, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average=average, zero_division=0)
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Model Comparison")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")
    
    return df


def plot_feature_importances_comparison(models, top_n=15):
    """
    Compare feature importances across Random Forest and XGBoost.
    
    Args:
        models: Dictionary of trained models
        top_n: Number of top features to display
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    if 'random_forest' in models:
        rf_imp = models['random_forest'].get_feature_importances(top_n=top_n, plot=False)
        top_rf = rf_imp.head(top_n)
        
        axes[0].barh(range(len(top_rf)), top_rf['importance'], color='steelblue')
        axes[0].set_yticks(range(len(top_rf)))
        axes[0].set_yticklabels(top_rf['feature'])
        axes[0].set_xlabel('Importance', fontsize=12)
        axes[0].set_title(f'Random Forest - Top {top_n} Features', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
    
    if 'xgboost' in models:
        xgb_imp = models['xgboost'].get_feature_importances(top_n=top_n, plot=False)
        top_xgb = xgb_imp.head(top_n)
        
        axes[1].barh(range(len(top_xgb)), top_xgb['importance'], color='darkorange')
        axes[1].set_yticks(range(len(top_xgb)))
        axes[1].set_yticklabels(top_xgb['feature'])
        axes[1].set_xlabel('Importance (Gain)', fontsize=12)
        axes[1].set_title(f'XGBoost - Top {top_n} Features', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.show()
