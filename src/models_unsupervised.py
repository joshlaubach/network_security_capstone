"""
models_unsupervised.py
----------------------
Unsupervised clustering models for anomaly detection on the BETH dataset.

Implements:
  - K-Means Clustering
  - DBSCAN (Density-Based Spatial Clustering)
  - Gaussian Mixture Models (GMM)

Target Variables:
  - 'sus': in-distribution outliers (suspicious but similar to training)
  - 'evil': out-of-distribution outliers (clearly malicious/different)

Strategy:
  Clustering algorithms are trained on normal data, then used to detect
  anomalies based on cluster assignments, distances, or probabilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
import warnings
warnings.filterwarnings('ignore')


# ================================================================
# Categorical Encoding Utilities  
# ================================================================

def encode_categoricals_for_clustering(X, encoder=None):
    """
    Encode categorical features for clustering (requires numeric input).
    Uses OneHot encoding to avoid false ordinal relationships.
    
    Args:
        X: Features (DataFrame)
        encoder: Pre-fitted encoder object (optional, for test data)
        
    Returns:
        Tuple of (encoded_X, encoder_object, categorical_cols)
    """
    if not isinstance(X, pd.DataFrame):
        return X, None, []
    
    # Auto-detect categorical columns
    categorical_cols = X.select_dtypes(include='object').columns.tolist()
    
    # No categorical columns to encode
    if len(categorical_cols) == 0:
        return X, None, []
    
    X_encoded = X.copy()
    
    # One-hot encoding for clustering (avoid false ordering)
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
    
    return X_encoded, encoder, categorical_cols


# ================================================================
# Feature Scaling Utilities
# ================================================================

def scale_features(X, method='standard', scaler=None):
    """
    Scale features using various methods.
    
    Args:
        X: Features to scale (DataFrame, array, or sparse matrix)
        method: Scaling method - 'standard', 'minmax', 'robust', or 'none'
        scaler: Pre-fitted scaler object (optional, for test data)
        
    Returns:
        Tuple of (scaled_X, scaler_object)
    """
    if method == 'none':
        return X, None
    
    # Import sparse utilities
    from scipy import sparse
    
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
    
    # Check if the values we'll be scaling are sparse (after DataFrame extraction)
    is_sparse = sparse.issparse(X_values)
    
    if scaler is None:
        # Fit new scaler
        if method == 'standard':
            # For sparse matrices, use with_mean=False to avoid densification
            scaler = StandardScaler(with_mean=not is_sparse)
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            # RobustScaler doesn't support sparse matrices well, convert to standard
            if is_sparse:
                print("Warning: RobustScaler not ideal for sparse data, using StandardScaler instead")
                scaler = StandardScaler(with_mean=False)
            else:
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


def compute_distance_matrix(X, metric='euclidean'):
    """
    Compute pairwise distance matrix using specified metric.
    
    Args:
        X: Feature matrix
        metric: Distance metric - 'euclidean', 'manhattan', 'cosine'
        
    Returns:
        Distance matrix
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if metric == 'euclidean':
        return euclidean_distances(X)
    elif metric == 'manhattan':
        return manhattan_distances(X)
    elif metric == 'cosine':
        return cosine_distances(X)
    else:
        raise ValueError(f"Unknown distance metric: {metric}")


# ================================================================
# Elbow Method and Silhouette Analysis
# ================================================================

def elbow_method(X, k_range=range(2, 15), random_state=42, plot=True):
    """
    Perform elbow method analysis for K-Means clustering.
    
    Args:
        X: Feature matrix
        k_range: Range of K values to test
        random_state: Random seed
        plot: Whether to display the elbow plot
        
    Returns:
        Dictionary with inertias, silhouette scores, and optimal k suggestion
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    inertias = []
    silhouette_scores = []
    k_values = list(k_range)
    
    print("\n" + "="*60)
    print("Elbow Method Analysis")
    print("="*60)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertia = kmeans.inertia_
        sil_score = silhouette_score(X, labels)
        
        inertias.append(inertia)
        silhouette_scores.append(sil_score)
        
        print(f"K={k:2d} | Inertia: {inertia:12.2f} | Silhouette: {sil_score:.4f}")
    
    # Find elbow using rate of change
    inertia_diffs = np.diff(inertias)
    elbow_idx = np.argmax(inertia_diffs[:-1] - inertia_diffs[1:]) + 1
    suggested_k_elbow = k_values[elbow_idx]
    
    # Best silhouette score
    best_sil_idx = np.argmax(silhouette_scores)
    suggested_k_silhouette = k_values[best_sil_idx]
    
    print("="*60)
    print(f"Suggested K (Elbow): {suggested_k_elbow}")
    print(f"Suggested K (Silhouette): {suggested_k_silhouette}")
    print("="*60 + "\n")
    
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Elbow plot
        ax1.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8)
        ax1.axvline(suggested_k_elbow, color='red', linestyle='--', 
                   label=f'Elbow at K={suggested_k_elbow}')
        ax1.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax1.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
        ax1.set_title('Elbow Method', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Silhouette plot
        ax2.plot(k_values, silhouette_scores, 'go-', linewidth=2, markersize=8)
        ax2.axvline(suggested_k_silhouette, color='red', linestyle='--',
                   label=f'Best at K={suggested_k_silhouette}')
        ax2.set_xlabel('Number of Clusters (K)', fontsize=12)
        ax2.set_ylabel('Silhouette Score', fontsize=12)
        ax2.set_title('Silhouette Analysis', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    return {
        'k_values': k_values,
        'inertias': inertias,
        'silhouette_scores': silhouette_scores,
        'suggested_k_elbow': suggested_k_elbow,
        'suggested_k_silhouette': suggested_k_silhouette
    }


def silhouette_analysis(X, k_range=range(2, 10), random_state=42, plot=True):
    """
    Detailed silhouette analysis with per-cluster visualization.
    
    Args:
        X: Feature matrix
        k_range: Range of K values to analyze
        random_state: Random seed
        plot: Whether to display silhouette plots
        
    Returns:
        Dictionary with silhouette scores and cluster-level details
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    results = {}
    k_values = list(k_range)
    
    print("\n" + "="*60)
    print("Detailed Silhouette Analysis")
    print("="*60)
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Overall silhouette score
        sil_avg = silhouette_score(X, labels)
        
        # Per-sample silhouette scores
        sample_silhouette_values = silhouette_samples(X, labels)
        
        # Per-cluster silhouette scores
        cluster_silhouettes = []
        for i in range(k):
            cluster_sil = sample_silhouette_values[labels == i].mean()
            cluster_silhouettes.append(cluster_sil)
        
        results[k] = {
            'avg_silhouette': sil_avg,
            'cluster_silhouettes': cluster_silhouettes,
            'sample_silhouettes': sample_silhouette_values,
            'labels': labels
        }
        
        print(f"K={k} | Avg Silhouette: {sil_avg:.4f} | "
              f"Min Cluster: {min(cluster_silhouettes):.4f} | "
              f"Max Cluster: {max(cluster_silhouettes):.4f}")
    
    print("="*60 + "\n")
    
    if plot and len(k_values) <= 6:
        n_plots = len(k_values)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_plots > 1 else [axes]
        
        for idx, k in enumerate(k_values):
            ax = axes[idx]
            sil_values = results[k]['sample_silhouettes']
            labels = results[k]['labels']
            avg_sil = results[k]['avg_silhouette']
            
            y_lower = 10
            for i in range(k):
                cluster_sil_values = sil_values[labels == i]
                cluster_sil_values.sort()
                
                size_cluster = cluster_sil_values.shape[0]
                y_upper = y_lower + size_cluster
                
                color = plt.cm.nipy_spectral(float(i) / k)
                ax.fill_betweenx(np.arange(y_lower, y_upper),
                                0, cluster_sil_values,
                                facecolor=color, edgecolor=color, alpha=0.7)
                
                ax.text(-0.05, y_lower + 0.5 * size_cluster, str(i))
                y_lower = y_upper + 10
            
            ax.set_title(f'K={k} (Avg: {avg_sil:.3f})', fontweight='bold')
            ax.set_xlabel('Silhouette Coefficient')
            ax.set_ylabel('Cluster')
            ax.axvline(x=avg_sil, color="red", linestyle="--", label='Average')
            ax.set_yticks([])
            ax.legend()
        
        # Hide unused subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return results


def compare_distance_metrics(X, k=8, metrics=['euclidean', 'manhattan', 'cosine'], 
                            random_state=42):
    """
    Compare K-Means clustering performance using different distance metrics.
    
    Note: sklearn's KMeans uses Euclidean by default. For other metrics,
    we compute distance matrices and analyze cluster quality.
    
    Args:
        X: Feature matrix
        k: Number of clusters
        metrics: List of distance metrics to compare
        random_state: Random seed
        
    Returns:
        DataFrame comparing metrics
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    print("\n" + "="*60)
    print(f"Distance Metric Comparison (K={k})")
    print("="*60)
    
    results = []
    
    # KMeans with Euclidean (default)
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels_euclidean = kmeans.fit_predict(X)
    
    for metric in metrics:
        if metric == 'euclidean':
            labels = labels_euclidean
            inertia = kmeans.inertia_
        else:
            # For non-euclidean metrics, we still use KMeans but evaluate with different metric
            labels = labels_euclidean  # Use same clustering
            
            # Compute custom inertia
            dist_matrix = compute_distance_matrix(X, metric)
            inertia = 0
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centroid = cluster_points.mean(axis=0, keepdims=True)
                    cluster_dists = compute_distance_matrix(
                        np.vstack([cluster_points, centroid]), 
                        metric
                    )[-1, :-1]
                    inertia += cluster_dists.sum()
        
        sil_score = silhouette_score(X, labels, metric=metric)
        
        results.append({
            'metric': metric,
            'inertia': inertia,
            'silhouette_score': sil_score
        })
        
        print(f"{metric:12s} | Inertia: {inertia:12.2f} | Silhouette: {sil_score:.4f}")
    
    print("="*60 + "\n")
    
    return pd.DataFrame(results)


# ================================================================
# K-Means Clustering for Anomaly Detection
# ================================================================

class KMeansAnomalyDetector:
    """
    K-Means clustering-based anomaly detector.
    
    Strategy:
      - Train K-Means on normal data
      - Anomalies are points far from their assigned cluster centers
      - Uses distance threshold (e.g., 95th percentile) to flag outliers
      
    MEMORY-EFFICIENT IMPLEMENTATION:
      - For large datasets (>100K), computes silhouette on subsample
      - Avoids O(n) memory for silhouette_score calculation
      - Reduces memory from ~58GB to 2GB for 763K samples
    """
    
    def __init__(self, n_clusters=8, contamination=0.05, random_state=42, 
                 distance_metric='euclidean', scaler='standard', 
                 max_silhouette_samples=100000):
        """
        Args:
            n_clusters: Number of clusters to create
            contamination: Expected proportion of outliers (for threshold)
            random_state: Random seed for reproducibility
            distance_metric: Distance metric for anomaly scoring
            scaler: Feature scaling method ('standard', 'minmax', 'robust', 'none')
            max_silhouette_samples: Maximum samples for silhouette calculation (memory limit)
        """
        self.n_clusters = n_clusters
        self.contamination = contamination
        self.random_state = random_state
        self.distance_metric = distance_metric
        self.scaler_method = scaler
        self.max_silhouette_samples = max_silhouette_samples
        self.model = None
        self.threshold = None
        self.scaler_obj = None
        self.encoder_obj = None
        self.categorical_cols = []
        self.silhouette_score_ = None
        
    def fit(self, X, y=None):
        """
        Fit K-Means on training data (assumed to be mostly normal).
        
        Args:
            X: Training features (DataFrame with categorical columns as strings, or sparse matrix)
            y: Not used (unsupervised), included for API consistency
        """
        # Import sparse utilities
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals_for_clustering(X)
        else:
            # X is already encoded (e.g., sparse matrix from TF-IDF)
            X_encoded = X
            self.encoder_obj = None
            self.categorical_cols = []
        
        # Apply feature scaling
        X_scaled, self.scaler_obj = scale_features(X_encoded, method=self.scaler_method)
        
        # Convert to dense array if needed (KMeans doesn't support sparse matrices well)
        if sparse.issparse(X_scaled):
            print(f"Converting sparse matrix ({X_scaled.shape}) to dense for KMeans...")
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        n_samples = len(X_scaled)
            
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(X_scaled)
        
        # Compute silhouette score on subsample if dataset is large
        # (silhouette_score has O(n) memory complexity)
        if n_samples > self.max_silhouette_samples:
            print(f"Dataset has {n_samples:,} samples (>{self.max_silhouette_samples:,})")
            print(f"       Computing silhouette on {self.max_silhouette_samples:,} subsample")
            print(f"       (prevents {n_samples**2/1e9:.1f}B distance calculations -> OOM)")
            
            np.random.seed(self.random_state)
            subsample_idx = np.random.choice(n_samples, self.max_silhouette_samples, replace=False)
            X_subsample = X_scaled[subsample_idx]
            labels_subsample = self.model.predict(X_subsample)
            self.silhouette_score_ = silhouette_score(X_subsample, labels_subsample)
        else:
            labels = self.model.labels_
            self.silhouette_score_ = silhouette_score(X_scaled, labels)
        
        # Compute distances to cluster centers for training data
        distances = self._compute_distances(X_scaled)
        
        # Set threshold at (1 - contamination) percentile
        self.threshold = np.percentile(distances, 100 * (1 - self.contamination))
        
        print(f"K-Means fitted with {self.n_clusters} clusters")
        print(f"Silhouette Score: {self.silhouette_score_:.4f}")
        print(f"Anomaly threshold set at {self.threshold:.4f}")
        print(f"Categorical cols encoded: {len(self.categorical_cols)}")
        print(f"Scaling: {self.scaler_method}, Distance: {self.distance_metric}")
        
        return self
    
    def _compute_distances(self, X):
        """
        Compute distance from each point to its NEAREST cluster center.
        
        CRITICAL: Uses model.transform() which computes distance to ALL cluster centers,
        then takes minimum. This is the correct anomaly score for K-Means clustering.
        
        Note: Only supports euclidean distance (sklearn KMeans default).
        For other metrics, use manual computation (commented out below).
        """
        from scipy import sparse
        
        # Convert sparse to dense if needed
        if sparse.issparse(X):
            X = X.toarray()
        
        # Use sklearn's optimized transform() method
        # Returns distances to ALL cluster centers, shape (n_samples, n_clusters)
        if self.distance_metric == 'euclidean':
            all_distances = self.model.transform(X)
            # Anomaly score = distance to NEAREST cluster center
            distances = np.min(all_distances, axis=1)
        else:
            # Fallback for non-euclidean metrics (slower)
            labels = self.model.predict(X)
            centers = self.model.cluster_centers_
            n_samples = X.shape[0]
            distances = np.zeros(n_samples)
            
            if self.distance_metric == 'manhattan':
                for i in range(n_samples):
                    distances[i] = np.sum(np.abs(X[i] - centers[labels[i]]))
            elif self.distance_metric == 'cosine':
                for i in range(n_samples):
                    dot_product = np.dot(X[i], centers[labels[i]])
                    norm_product = np.linalg.norm(X[i]) * np.linalg.norm(centers[labels[i]])
                    distances[i] = 1 - (dot_product / (norm_product + 1e-10))
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        return distances
    
    def predict(self, X, threshold=None):
        """
        Predict anomaly labels: 0 = normal, 1 = anomaly.
        
        Args:
            X: Features to predict on (DataFrame, array, or sparse matrix)
            threshold (float, optional): A specific distance threshold to use. 
                                         If None, uses the one from `fit`.
            
        Returns:
            Binary labels (0/1)
        """
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, _, _ = encode_categoricals_for_clustering(
                X, encoder=self.encoder_obj
            )
        else:
            X_encoded = X
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        # Convert to dense if needed
        if sparse.issparse(X_scaled):
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        distances = self._compute_distances(X_scaled)
        
        # Use the provided threshold, or the one from fitting if None
        active_threshold = threshold if threshold is not None else self.threshold
        
        predictions = (distances > active_threshold).astype(int)
        
        return predictions
    
    def decision_function(self, X):
        """
        Return anomaly scores (higher = more anomalous).
        
        Args:
            X: Features to score (DataFrame, array, or sparse matrix)
            
        Returns:
            Anomaly scores (distance from cluster center)
        """
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, _, _ = encode_categoricals_for_clustering(
                X, encoder=self.encoder_obj
            )
        else:
            X_encoded = X
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        # Convert to dense if needed
        if sparse.issparse(X_scaled):
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        return self._compute_distances(X_scaled)
    
    def get_cluster_info(self, X):
        """
        Get cluster assignments and distances for analysis.
        
        Returns:
            Dictionary with cluster labels, distances, and statistics
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        labels = self.model.predict(X_scaled)
        distances = self._compute_distances(X_scaled)
        
        return {
            'labels': labels,
            'distances': distances,
            'n_clusters': self.n_clusters,
            'cluster_sizes': np.bincount(labels),
            'threshold': self.threshold,
            'silhouette_score': self.silhouette_score_,
            'distance_metric': self.distance_metric
        }


# ================================================================
# DBSCAN for Anomaly Detection
# ================================================================

class DBSCANAnomalyDetector:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    
    Strategy:
      - Groups dense regions into clusters
      - Points that don't belong to any cluster are marked as noise (-1)
      - Noise points are considered anomalies
      
    MEMORY-EFFICIENT IMPLEMENTATION:
      - For large datasets (>100K), trains on a subsample to avoid O(n) memory
      - Uses NearestNeighbors for prediction on remaining data
      - Reduces memory from ~58GB to 2-8GB for 763K samples
    """
    
    def __init__(self, eps=0.5, min_samples=5, metric='euclidean', scaler='standard', 
                 max_train_samples=100000, random_state=42):
        """
        Args:
            eps: Maximum distance between two samples to be neighbors
            min_samples: Minimum number of samples in a neighborhood for core point
            metric: Distance metric to use
            scaler: Feature scaling method ('standard', 'minmax', 'robust', 'none')
            max_train_samples: Maximum samples for DBSCAN fit (memory limit)
            random_state: Random seed for sampling
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.scaler_method = scaler
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.model = None
        self.core_sample_indices = None
        self.scaler_obj = None
        self.encoder_obj = None
        self.categorical_cols = []
        self.X_train_scaled = None  # Store training data for prediction
        self.train_labels = None    # Store training labels
        
    def fit(self, X, y=None):
        """
        Fit DBSCAN on training data (with memory-efficient subsampling).
        
        Args:
            X: Training features (DataFrame with categorical columns as strings, or sparse matrix)
            y: Not used (unsupervised)
        """
        # Import sparse utilities
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals_for_clustering(X)
        else:
            # X is already encoded (e.g., sparse matrix from TF-IDF)
            X_encoded = X
            self.encoder_obj = None
            self.categorical_cols = []
        
        # Apply feature scaling
        X_scaled, self.scaler_obj = scale_features(X_encoded, method=self.scaler_method)
        
        # Convert to dense array if needed (DBSCAN doesn't support sparse matrices well)
        if sparse.issparse(X_scaled):
            print(f"Converting sparse matrix ({X_scaled.shape}) to dense for DBSCAN...")
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        n_samples = len(X_scaled)
        
        # Memory-efficient: subsample if dataset is too large
        if n_samples > self.max_train_samples:
            print(f"Dataset has {n_samples:,} samples (>{self.max_train_samples:,})")
            print(f"       Subsampling to {self.max_train_samples:,} for DBSCAN training")
            print(f"       (prevents {n_samples**2/1e9:.1f}B distance calculations -> OOM)")
            
            np.random.seed(self.random_state)
            subsample_idx = np.random.choice(n_samples, self.max_train_samples, replace=False)
            X_train_subsample = X_scaled[subsample_idx]
        else:
            X_train_subsample = X_scaled
            subsample_idx = np.arange(n_samples)
        
        # Store full training data for prediction
        self.X_train_scaled = X_scaled
        
        # Fit DBSCAN on subsample
        self.model = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric
        )
        
        labels_subsample = self.model.fit_predict(X_train_subsample)
        self.core_sample_indices = self.model.core_sample_indices_
        
        # Store labels for full dataset (subsample labels + predict remaining)
        if n_samples > self.max_train_samples:
            # Use NearestNeighbors to label remaining samples
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
            nn.fit(X_train_subsample)
            
            # Get remaining samples
            remaining_idx = np.setdiff1d(np.arange(n_samples), subsample_idx)
            X_remaining = X_scaled[remaining_idx]
            
            # Find nearest neighbor in subsample for each remaining point
            distances, indices = nn.kneighbors(X_remaining)
            
            # Assign labels based on nearest neighbor
            full_labels = np.zeros(n_samples, dtype=int)
            full_labels[subsample_idx] = labels_subsample
            full_labels[remaining_idx] = labels_subsample[indices.flatten()]
            
            # Mark far points as noise
            far_points = distances.flatten() > self.eps
            full_labels[remaining_idx[far_points]] = -1
            
            self.train_labels = full_labels
        else:
            self.train_labels = labels_subsample
        
        n_clusters = len(set(self.train_labels)) - (1 if -1 in self.train_labels else 0)
        n_noise = list(self.train_labels).count(-1)
        
        print(f"DBSCAN fitted:")
        print(f"       Clusters found: {n_clusters}")
        print(f"       Noise points: {n_noise:,} ({100*n_noise/n_samples:.2f}%)")
        print(f"       Trained on: {len(X_train_subsample):,}/{n_samples:,} samples")
        print(f"       Categorical cols encoded: {len(self.categorical_cols)}")
        print(f"       Scaling: {self.scaler_method}, Distance: {self.metric}")
        
        return self
    
    def predict(self, X, threshold=None):
        """
        Predict anomaly labels for new data (memory-efficient).
        
        Strategy:
          - Use NearestNeighbors to find closest training sample
          - If nearest neighbor is noise or far away, mark as anomaly
        
        Args:
            X: Features to predict on (DataFrame, array, or sparse matrix)
            threshold (float, optional): A specific distance threshold (eps) to use.
                                         If None, uses the one from `__init__`.
            
        Returns:
            Binary labels (0 = normal, 1 = anomaly)
        """
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, _, _ = encode_categoricals_for_clustering(
                X, encoder=self.encoder_obj
            )
        else:
            X_encoded = X
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        # Convert to dense if needed
        if sparse.issparse(X_scaled):
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        # Use NearestNeighbors instead of fit_predict (memory efficient)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(self.X_train_scaled)
        
        distances, indices = nn.kneighbors(X_scaled)
        
        # Get labels of nearest neighbors
        nearest_labels = self.train_labels[indices.flatten()]
        
        # Use the provided threshold (eps), or the one from initialization if None
        active_threshold = threshold if threshold is not None else self.eps
        
        # Mark as anomaly if:
        # 1. Nearest neighbor is noise (-1), OR
        # 2. Distance to nearest neighbor > active_threshold
        n_samples = X_scaled.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        predictions[(nearest_labels == -1) | (distances.flatten() > active_threshold)] = 1
        
        return predictions
    
    def decision_function(self, X):
        """
        Return anomaly scores based on distance to nearest cluster (memory-efficient).
        
        CRITICAL FIX: Returns raw distance to nearest neighbor, NOT normalized by eps.
        This provides a continuous anomaly score where higher = more anomalous.
        
        Returns:
            Scores where points far from clusters get high values
        """
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, _, _ = encode_categoricals_for_clustering(
                X, encoder=self.encoder_obj
            )
        else:
            X_encoded = X
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        # Convert to dense if needed
        if sparse.issparse(X_scaled):
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        # Use NearestNeighbors for scoring (memory efficient)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(self.X_train_scaled)
        
        distances, indices = nn.kneighbors(X_scaled)
        
        # Return raw distances as anomaly scores
        # Higher distance = more anomalous
        # Do NOT normalize by eps or cap at 1.0 - we need continuous scores for threshold optimization
        return distances.flatten()
    
    def get_cluster_info(self, X):
        """
        Get cluster assignments and statistics (memory-efficient).
        """
        # Use predict method instead of fit_predict
        predictions = self.predict(X)
        
        return {
            'labels': self.train_labels if X is self.X_train_scaled else predictions,
            'n_clusters': len(set(self.train_labels)) - (1 if -1 in self.train_labels else 0),
            'n_noise': list(self.train_labels).count(-1),
            'noise_rate': list(self.train_labels).count(-1) / len(self.train_labels)
        }


# ================================================================
# Gaussian Mixture Model (GMM) for Anomaly Detection
# ================================================================

class GMManomalyDetector:
    """
    Gaussian Mixture Model for anomaly detection.
    
    Strategy:
      - Fit a mixture of Gaussian distributions to normal data
      - Anomalies have low probability under the fitted model
      - Use log-likelihood threshold to identify outliers
      
    MEMORY-EFFICIENT IMPLEMENTATION:
      - For large datasets (>100K), trains on a subsample to avoid memory issues
      - GMM with 'full' covariance can use significant memory during EM iterations
      - Reduces memory usage while maintaining model quality
    """
    
    def __init__(self, n_components=3, contamination=0.05, random_state=42, 
                 covariance_type='full', scaler='standard', max_train_samples=100000):
        """
        Args:
            n_components: Number of Gaussian components
            contamination: Expected proportion of outliers
            random_state: Random seed
            covariance_type: Type of covariance ('full', 'tied', 'diag', 'spherical')
            scaler: Feature scaling method ('standard', 'minmax', 'robust', 'none')
            max_train_samples: Maximum samples for GMM fit (memory limit)
        """
        self.n_components = n_components
        self.contamination = contamination
        self.random_state = random_state
        self.covariance_type = covariance_type
        self.scaler_method = scaler
        self.max_train_samples = max_train_samples
        self.model = None
        self.threshold = None
        self.scaler_obj = None
        self.encoder_obj = None
        self.categorical_cols = []
        
    def fit(self, X, y=None):
        """
        Fit GMM on training data (with memory-efficient subsampling).
        
        Args:
            X: Training features (DataFrame with categorical columns as strings, or sparse matrix)
            y: Not used (unsupervised)
        """
        # Import sparse utilities
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals_for_clustering(X)
        else:
            # X is already encoded (e.g., sparse matrix from TF-IDF)
            X_encoded = X
            self.encoder_obj = None
            self.categorical_cols = []
        
        # Apply feature scaling
        X_scaled, self.scaler_obj = scale_features(X_encoded, method=self.scaler_method)
        
        # Convert to dense array if needed (GMM doesn't support sparse matrices)
        if sparse.issparse(X_scaled):
            print(f"Converting sparse matrix ({X_scaled.shape}) to dense for GMM...")
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        n_samples = len(X_scaled)
        
        # Memory-efficient: subsample if dataset is too large
        # GMM with 'full' covariance computes n_features x n_features covariance matrices
        # for each component during EM iterations (memory intensive)
        if n_samples > self.max_train_samples:
            print(f"Dataset has {n_samples:,} samples (>{self.max_train_samples:,})")
            print(f"       Subsampling to {self.max_train_samples:,} for GMM training")
            print(f"       (reduces EM iteration memory for '{self.covariance_type}' covariance)")
            
            np.random.seed(self.random_state)
            subsample_idx = np.random.choice(n_samples, self.max_train_samples, replace=False)
            X_train_subsample = X_scaled[subsample_idx]
        else:
            X_train_subsample = X_scaled
        
        # Fit GMM on subsample (or full data if small enough)
        self.model = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            random_state=self.random_state,
            n_init=10
        )
        
        self.model.fit(X_train_subsample)
        
        # Compute log-likelihood scores for FULL training data
        # (score_samples is O(n) not O(n), so safe to use full dataset)
        log_likelihoods = self.model.score_samples(X_scaled)
        
        # Set threshold at contamination percentile
        # Lower log-likelihood = more anomalous
        self.threshold = np.percentile(log_likelihoods, 100 * self.contamination)
        
        print(f"GMM fitted with {self.n_components} components")
        print(f"Trained on: {len(X_train_subsample):,}/{n_samples:,} samples")
        print(f"Log-likelihood threshold: {self.threshold:.4f}")
        print(f"Covariance type: {self.covariance_type}")
        print(f"Categorical cols encoded: {len(self.categorical_cols)}")
        print(f"Scaling: {self.scaler_method}")
        
        # Check convergence
        if self.model.converged_:
            print(f"GMM converged in {self.model.n_iter_} iterations")
        else:
            print(f"[WARN] GMM did not converge!")
        
        return self
    
    def predict(self, X, threshold=None):
        """
        Predict anomaly labels: 0 = normal, 1 = anomaly.
        
        Args:
            X: Features to predict on (DataFrame, array, or sparse matrix)
            threshold (float, optional): A specific log-likelihood threshold to use.
                                         If None, uses the one from `fit`.
            
        Returns:
            Binary labels
        """
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, _, _ = encode_categoricals_for_clustering(
                X, encoder=self.encoder_obj
            )
        else:
            X_encoded = X
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        # Convert to dense if needed
        if sparse.issparse(X_scaled):
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        log_likelihoods = self.model.score_samples(X_scaled)
        
        # Use the provided threshold, or the one from fitting if None
        active_threshold = threshold if threshold is not None else self.threshold
        
        # Points with log-likelihood below threshold are anomalies
        predictions = (log_likelihoods < active_threshold).astype(int)
        
        return predictions
    
    def decision_function(self, X):
        """
        Return anomaly scores (negative log-likelihood).
        Higher scores = more anomalous.
        
        Args:
            X: Features to score (DataFrame, array, or sparse matrix)
            
        Returns:
            Anomaly scores
        """
        from scipy import sparse
        
        # Only encode categoricals if X is a DataFrame (not already a sparse matrix)
        if isinstance(X, pd.DataFrame):
            X_encoded, _, _ = encode_categoricals_for_clustering(
                X, encoder=self.encoder_obj
            )
        else:
            X_encoded = X
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        # Convert to dense if needed
        if sparse.issparse(X_scaled):
            X_scaled = X_scaled.toarray()
        elif isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        # Return negative log-likelihood so higher = more anomalous
        log_likelihoods = self.model.score_samples(X_scaled)
        return -log_likelihoods
    
    def predict_proba(self, X):
        """
        Get probability of belonging to each Gaussian component.
        
        Args:
            X: Features (with categorical columns as strings)
            
        Returns:
            Probability matrix (n_samples, n_components)
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        return self.model.predict_proba(X_scaled)
    
    def get_cluster_info(self, X):
        """
        Get component assignments and statistics.
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        labels = self.model.predict(X_scaled)
        log_likelihoods = self.model.score_samples(X_scaled)
        
        return {
            'labels': labels,
            'log_likelihoods': log_likelihoods,
            'n_components': self.n_components,
            'component_sizes': np.bincount(labels),
            'threshold': self.threshold,
            'weights': self.model.weights_,
            'converged': self.model.converged_,
            'covariance_type': self.covariance_type
        }


# ================================================================
# Utility Functions
# ================================================================

def train_all_models(X_train, contamination=0.05, random_state=42, 
                    scaler='standard', distance_metric='euclidean'):
    """
    Train all three clustering models on the same training data.
    
    Args:
        X_train: Training features (pandas DataFrame or numpy array)
        contamination: Expected outlier rate
        random_state: Random seed
        scaler: Feature scaling method ('standard', 'minmax', 'robust', 'none')
        distance_metric: Distance metric for K-Means ('euclidean', 'manhattan', 'cosine')
        
    Returns:
        Dictionary of trained models
    """
    print("\n" + "="*60)
    print("Training All Unsupervised Models")
    print("="*60)
    
    models = {}
    
    # K-Means
    print("\n[1/3] Training K-Means...")
    kmeans = KMeansAnomalyDetector(
        n_clusters=8,
        contamination=contamination,
        random_state=random_state,
        distance_metric=distance_metric,
        scaler=scaler
    )
    kmeans.fit(X_train)
    models['kmeans'] = kmeans
    
    # DBSCAN
    print("\n[2/3] Training DBSCAN...")
    dbscan = DBSCANAnomalyDetector(
        eps=0.5,
        min_samples=5,
        metric=distance_metric,
        scaler=scaler
    )
    dbscan.fit(X_train)
    models['dbscan'] = dbscan
    
    # GMM
    print("\n[3/3] Training GMM...")
    gmm = GMManomalyDetector(
        n_components=5,
        contamination=contamination,
        random_state=random_state,
        scaler=scaler
    )
    gmm.fit(X_train)
    models['gmm'] = gmm
    
    print("\n" + "="*60)
    print("All models trained successfully!")
    print("="*60 + "\n")
    
    return models


def get_predictions_all_models(models, X_test, thresholds=None):
    """
    Get predictions from all trained models, using optimized thresholds if provided.
    
    Args:
        models (dict): Dictionary of trained models.
        X_test: Test features.
        thresholds (dict, optional): Dictionary of optimized thresholds, 
                                     e.g., {'kmeans': 1.23, 'gmm': -4.56}.
        
    Returns:
        Dictionary of predictions and scores.
    """
    results = {}
    thresholds = thresholds or {}
    
    for name, model in models.items():
        # Get the specific threshold for this model, or None if not provided
        model_threshold = thresholds.get(name)
        
        results[name] = {
            'predictions': model.predict(X_test, threshold=model_threshold),
            'scores': model.decision_function(X_test)
        }
    
    return results


def analyze_sus_vs_evil(models, X_test, y_sus, y_evil, thresholds=None):
    """
    Analyze how well each model distinguishes 'sus' vs 'evil' outliers.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_sus: Binary labels for 'sus' (in-distribution outliers)
        y_evil: Binary labels for 'evil' (out-of-distribution outliers)
        thresholds (dict, optional): Dictionary of optimized thresholds to use for prediction.
        
    Returns:
        DataFrame with detection rates for each model and outlier type
    """
    results = []
    thresholds = thresholds or {}
    
    for name, model in models.items():
        # Get the specific threshold for this model, or None if not provided
        model_threshold = thresholds.get(name)
        
        preds = model.predict(X_test, threshold=model_threshold)
        scores = model.decision_function(X_test)
        
        # Convert to numpy if needed
        if isinstance(y_sus, pd.Series):
            y_sus = y_sus.values
        if isinstance(y_evil, pd.Series):
            y_evil = y_evil.values
            
        # Detection rates
        sus_detected = preds[y_sus == 1].sum() if (y_sus == 1).sum() > 0 else 0
        sus_total = (y_sus == 1).sum()
        sus_rate = sus_detected / sus_total if sus_total > 0 else 0
        
        evil_detected = preds[y_evil == 1].sum() if (y_evil == 1).sum() > 0 else 0
        evil_total = (y_evil == 1).sum()
        evil_rate = evil_detected / evil_total if evil_total > 0 else 0
        
        # Average scores for each group
        normal_mask = (y_sus == 0) & (y_evil == 0)
        sus_mask = (y_sus == 1) & (y_evil == 0)
        evil_mask = (y_evil == 1)
        
        avg_score_normal = scores[normal_mask].mean() if normal_mask.sum() > 0 else 0
        avg_score_sus = scores[sus_mask].mean() if sus_mask.sum() > 0 else 0
        avg_score_evil = scores[evil_mask].mean() if evil_mask.sum() > 0 else 0
        
        results.append({
            'model': name,
            'sus_detection_rate': sus_rate,
            'evil_detection_rate': evil_rate,
            'avg_score_normal': avg_score_normal,
            'avg_score_sus': avg_score_sus,
            'avg_score_evil': avg_score_evil
        })
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print("Sus vs Evil Detection Analysis")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60 + "\n")
    
    return df


def optimize_threshold_for_target(scores, y_true, model_name, lower_bound, upper_bound, metric='f1', target_name='sus'):
    """
    Find the optimal threshold that maximizes a given metric for a specific target.
    
    Args:
        scores (np.array): Anomaly scores where higher is more anomalous.
        y_true (np.array): True binary labels (0 or 1).
        model_name (str): Name of the model for display.
        lower_bound (float): The lower bound for the threshold search.
        upper_bound (float): The upper bound for the threshold search.
        metric (str): The metric to optimize ('f1', 'precision', 'recall', 'accuracy').
        target_name (str): Name of the target for display purposes.
        
    Returns:
        dict: A dictionary containing the best threshold and associated metrics.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
    
    best_metric_val = -1
    best_threshold = -1
    best_metrics = {}

    # Use a fixed number of thresholds (100) for efficiency
    # This is much faster than testing every integer in the range
    n_thresholds = 100
    thresholds = np.linspace(lower_bound, upper_bound, n_thresholds)
    
    print(f"Optimizing '{model_name}' for '{target_name}': testing {len(thresholds)} thresholds...")

    for threshold in thresholds:
        y_pred = (scores > threshold).astype(int)
        
        # Ensure there are both predicted classes to avoid errors
        if len(np.unique(y_pred)) < 2:
            continue
        
        # Calculate all metrics
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        metric_map = {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        }
        
        current_metric_val = metric_map.get(metric)
        
        if current_metric_val > best_metric_val:
            best_metric_val = current_metric_val
            best_threshold = threshold
            best_metrics = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'accuracy': accuracy,
                'threshold': threshold
            }

    if not best_metrics:
        print(f"Warning: Could not find an optimal threshold for {model_name} on {target_name}.")
        return {
            'f1': 0, 'precision': 0, 'recall': 0, 'accuracy': 0, 'threshold': np.mean(scores)
        }
        
    print(f"\nOptimal threshold for '{model_name}' on '{target_name}' (optimized for '{metric}'):")
    print(f"  Threshold: {best_metrics['threshold']:.4f}")
    print(f"  F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall:    {best_metrics['recall']:.4f}")
    print(f"  Accuracy:  {best_metrics['accuracy']:.4f}")

    return best_metrics


def analyze_threshold_performance(scores, y_true, model_name, val_optimized_threshold=None):
    """
    Performs a comprehensive analysis of anomaly score thresholds and visualizes performance.

    Args:
        scores (np.array): Anomaly scores where higher is more anomalous.
        y_true (np.array): True binary labels (0 or 1).
        model_name (str): Name of the model for plotting titles.
        val_optimized_threshold (float, optional): A pre-optimized threshold (e.g., from a validation set)
                                                   to compare against. Defaults to None.
    """
    from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

    print("="*80)
    print(f"COMPREHENSIVE THRESHOLD ANALYSIS - {model_name.upper()}")
    print("="*80)

    # Calculate metrics across a range of thresholds
    threshold_range = np.linspace(np.percentile(scores, 1), np.percentile(scores, 99), 200)
    
    results_detailed = []
    for threshold in threshold_range:
        y_pred = (scores > threshold).astype(int)
        
        if len(np.unique(y_pred)) < 2:
            continue

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        results_detailed.append({
            'threshold': threshold,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn
        })

    results_df = pd.DataFrame(results_detailed)
    if results_df.empty:
        print(f"Could not generate performance analysis for {model_name}. All thresholds resulted in single-class predictions.")
        return

    # Find best F1-score threshold from the detailed analysis
    best_f1_idx = results_df['f1'].idxmax()
    best_f1_threshold = results_df.loc[best_f1_idx, 'threshold']

    print(f"\n[{model_name.upper()} THRESHOLD RECOMMENDATIONS]")
    print(f"\n1. Best F1-Score Threshold: {best_f1_threshold:.2f}")
    print(f"   F1:        {results_df.loc[best_f1_idx, 'f1']:.4f}")
    print(f"   Precision: {results_df.loc[best_f1_idx, 'precision']:.4f}")
    print(f"   Recall:    {results_df.loc[best_f1_idx, 'recall']:.4f}")
    print(f"   FPR:       {results_df.loc[best_f1_idx, 'fpr']:.4f}")

    if val_optimized_threshold:
        y_pred_val = (scores > val_optimized_threshold).astype(int)
        f1_val = f1_score(y_true, y_pred_val, zero_division=0)
        recall_val = recall_score(y_true, y_pred_val, zero_division=0)
        print(f"\n2. Validation-Optimized Threshold: {val_optimized_threshold:.2f}")
        print(f"   (Performance on this data: F1={f1_val:.4f}, Recall={recall_val:.4f})")

    # Create 4-panel visualization
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    axs = axs.flatten()
    fig.suptitle(f'Threshold Performance Analysis for {model_name}', fontsize=16, fontweight='bold')

    # Plot 1: Precision, Recall, F1-Score vs Threshold
    axs[0].plot(results_df['threshold'], results_df['precision'], label='Precision', color='blue', linewidth=2)
    axs[0].plot(results_df['threshold'], results_df['recall'], label='Recall', color='green', linewidth=2)
    axs[0].plot(results_df['threshold'], results_df['f1'], label='F1-Score', color='red', linewidth=2)
    axs[0].axvline(x=best_f1_threshold, color='red', linestyle='--', alpha=0.7, label=f'Best F1: {best_f1_threshold:.1f}')
    if val_optimized_threshold:
        axs[0].axvline(x=val_optimized_threshold, color='orange', linestyle='--', alpha=0.7, label=f'Val-Opt: {val_optimized_threshold:.1f}')
    axs[0].set_xlabel('Threshold', fontsize=12)
    axs[0].set_ylabel('Score', fontsize=12)
    axs[0].set_title('Precision, Recall, F1 vs Threshold', fontsize=14)
    axs[0].legend(fontsize=10)
    axs[0].grid(True, alpha=0.3)

    # Plot 2: Accuracy and FPR vs Threshold
    ax2 = axs[1]
    ax2.plot(results_df['threshold'], results_df['accuracy'], label='Accuracy', color='purple', linewidth=2)
    ax2.set_xlabel('Threshold', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12, color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')
    ax2.axvline(x=best_f1_threshold, color='red', linestyle='--', alpha=0.7)
    ax2_twin = ax2.twinx()
    ax2_twin.plot(results_df['threshold'], results_df['fpr'] * 100, label='FPR', color='orange', linewidth=2)
    ax2_twin.set_ylabel('False Positive Rate (%)', fontsize=12, color='orange')
    ax2_twin.tick_params(axis='y', labelcolor='orange')
    ax2.set_title('Accuracy and FPR vs Threshold', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Plot 3: True/False Positives vs Threshold
    axs[2].plot(results_df['threshold'], results_df['tp'], label='True Positives', color='green', linewidth=2)
    axs[2].plot(results_df['threshold'], results_df['fp'], label='False Positives', color='red', linewidth=2)
    axs[2].axvline(x=best_f1_threshold, color='red', linestyle='--', alpha=0.7)
    axs[2].set_xlabel('Threshold', fontsize=12)
    axs[2].set_ylabel('Count', fontsize=12)
    axs[2].set_title('TP and FP Counts vs Threshold', fontsize=14)
    axs[2].legend(fontsize=10)
    axs[2].grid(True, alpha=0.3)

    # Plot 4: Score Distribution with Threshold Lines
    normal_scores = scores[y_true == 0]
    anomaly_scores = scores[y_true == 1]
    axs[3].hist(normal_scores, bins=100, alpha=0.6, label='Normal', color='green', density=True)
    axs[3].hist(anomaly_scores, bins=100, alpha=0.6, label='Anomaly', color='red', density=True)
    axs[3].axvline(x=best_f1_threshold, color='red', linestyle='-', linewidth=2, label=f'Best F1: {best_f1_threshold:.1f}')
    if val_optimized_threshold:
        axs[3].axvline(x=val_optimized_threshold, color='orange', linestyle='--', linewidth=2, label=f'Val-Opt: {val_optimized_threshold:.1f}')
    axs[3].set_xlabel('Anomaly Score', fontsize=12)
    axs[3].set_ylabel('Density', fontsize=12)
    axs[3].set_title('Score Distribution with Thresholds', fontsize=14)
    axs[3].set_xlim([results_df['threshold'].min(), results_df['threshold'].max()])
    axs[3].legend(fontsize=10)
    axs[3].grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
    print("\n" + "="*80)
    
    return results_df


def plot_roc_curve_analysis(y_true, scores, model_name, f1_opt_threshold, f1_opt_fpr, f1_opt_recall):
    """
    Analyzes and visualizes the ROC curve for anomaly detection performance.

    Args:
        y_true (np.array): True binary labels.
        scores (np.array): Anomaly scores where higher is more anomalous.
        model_name (str): Name of the model for titles.
        f1_opt_threshold (float): The threshold previously optimized for the best F1-score.
        f1_opt_fpr (float): The False Positive Rate at the F1-optimized threshold.
        f1_opt_recall (float): The True Positive Rate (Recall) at the F1-optimized threshold.
    """
    from sklearn.metrics import roc_curve, roc_auc_score

    print("="*80)
    print(f"ROC CURVE ANALYSIS - {model_name.upper()}")
    print("="*80)

    # Compute ROC curve and AUC score
    fpr_roc, tpr_roc, thresholds_roc = roc_curve(y_true, scores)
    auc_score = roc_auc_score(y_true, scores)

    print(f"\n[ROC-AUC Analysis]")
    print(f"  AUC Score: {auc_score:.4f}")
    print(f"  Interpretation: {'Excellent' if auc_score > 0.9 else 'Good' if auc_score > 0.8 else 'Fair'} discrimination")

    # Find optimal threshold on ROC curve (Youden's J statistic)
    j_scores = tpr_roc - fpr_roc
    optimal_idx = np.argmax(j_scores)
    optimal_roc_threshold = thresholds_roc[optimal_idx]
    optimal_tpr = tpr_roc[optimal_idx]
    optimal_fpr = fpr_roc[optimal_idx]

    print(f"\n[Optimal Threshold by ROC (Youden's J)]")
    print(f"  Threshold: {optimal_roc_threshold:.2f}")
    print(f"  TPR (Recall): {optimal_tpr:.4f}")
    print(f"  FPR: {optimal_fpr:.4f}")

    print(f"\n[Comparison: F1-Optimal vs. ROC-Optimal]")
    print(f"  F1-Optimal (Threshold={f1_opt_threshold:.2f}):")
    print(f"    - Recall: {f1_opt_recall:.4f}, FPR: {f1_opt_fpr:.4f}")
    print(f"  ROC-Optimal (Threshold={optimal_roc_threshold:.2f}):")
    print(f"    - Recall: {optimal_tpr:.4f}, FPR: {optimal_fpr:.4f}")

    # Visualize ROC Curve
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    ax.plot(fpr_roc, tpr_roc, color='blue', lw=2, label=f'{model_name} ROC (AUC = {auc_score:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
    ax.scatter([optimal_fpr], [optimal_tpr], color='red', s=200, marker='*', 
               edgecolor='black', linewidth=1.5, zorder=5, 
               label=f'ROC-Optimal (Thresh={optimal_roc_threshold:.1f})')
    ax.scatter([f1_opt_fpr], [f1_opt_recall], color='orange', s=150, marker='D',
               edgecolor='black', linewidth=1.5, zorder=5,
               label=f'F1-Optimal (Thresh={f1_opt_threshold:.1f})')
    
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate (Recall)', fontsize=13, fontweight='bold')
    ax.set_title(f'ROC Curve: {model_name} Anomaly Detection', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.4)
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])

    plt.tight_layout()
    plt.show()
    print("\n" + "="*80)


def tune_dbscan(X, eps_range=[0.5, 1.0, 1.5, 2.0], min_samples_range=[5, 10, 15, 20], 
                metric='euclidean', plot=True):
    """
    Tune DBSCAN hyperparameters using silhouette score and cluster statistics.
    
    Args:
        X: Feature matrix (can be sparse)
        eps_range: List of epsilon values to try
        min_samples_range: List of min_samples values to try
        metric: Distance metric to use
        plot: Whether to display visualization
        
    Returns:
        Dictionary with tuning results and best parameters
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Get number of samples (works for both sparse and dense)
    n_samples = X.shape[0]
    
    print("\n" + "="*60)
    print("DBSCAN Hyperparameter Tuning")
    print("="*60)
    
    results = []
    
    for eps in eps_range:
        for min_samples in min_samples_range:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
            labels = dbscan.fit_predict(X)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = list(labels).count(-1)
            
            # Only compute silhouette if we have valid clusters
            if n_clusters > 1 and n_clusters < n_samples - 1:
                # Filter out noise points for silhouette calculation
                mask = labels != -1
                if mask.sum() > 0:
                    try:
                        sil_score = silhouette_score(X[mask], labels[mask])
                    except:
                        sil_score = -1
                else:
                    sil_score = -1
            else:
                sil_score = -1
            
            results.append({
                'eps': eps,
                'min_samples': min_samples,
                'n_clusters': n_clusters,
                'n_noise': n_noise,
                'noise_ratio': n_noise / n_samples,
                'silhouette': sil_score
            })
            
            print(f"eps={eps:.1f}, min_samples={min_samples:2d} | "
                  f"Clusters: {n_clusters:2d} | Noise: {n_noise:5d} ({100*n_noise/n_samples:5.1f}%) | "
                  f"Silhouette: {sil_score:6.3f}")
    
    results_df = pd.DataFrame(results)
    
    # Find best parameters (prioritize silhouette, then minimize noise ratio)
    valid_results = results_df[results_df['silhouette'] > 0]
    
    if len(valid_results) > 0:
        # Best by silhouette score
        best_idx = valid_results['silhouette'].idxmax()
        best_params = results_df.loc[best_idx]
        
        print("\n" + "="*60)
        print("Best Parameters (by Silhouette Score):")
        print(f"  eps = {best_params['eps']}")
        print(f"  min_samples = {int(best_params['min_samples'])}")
        print(f"  n_clusters = {int(best_params['n_clusters'])}")
        print(f"  noise_ratio = {best_params['noise_ratio']:.3f}")
        print(f"  silhouette = {best_params['silhouette']:.3f}")
        print("="*60 + "\n")
    else:
        print("\n  Warning: No valid clustering found. Try adjusting parameter ranges.")
        best_params = results_df.iloc[0]
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pivot for heatmap
        pivot_sil = results_df.pivot(index='eps', columns='min_samples', values='silhouette')
        pivot_noise = results_df.pivot(index='eps', columns='min_samples', values='noise_ratio')
        
        # Plot 1: Silhouette scores
        im1 = axes[0].imshow(pivot_sil.values, cmap='viridis', aspect='auto')
        axes[0].set_xticks(range(len(pivot_sil.columns)))
        axes[0].set_yticks(range(len(pivot_sil.index)))
        axes[0].set_xticklabels(pivot_sil.columns)
        axes[0].set_yticklabels(pivot_sil.index)
        axes[0].set_xlabel('min_samples')
        axes[0].set_ylabel('eps')
        axes[0].set_title('Silhouette Score', fontweight='bold')
        plt.colorbar(im1, ax=axes[0])
        
        # Plot 2: Noise ratio
        im2 = axes[1].imshow(pivot_noise.values, cmap='RdYlGn_r', aspect='auto')
        axes[1].set_xticks(range(len(pivot_noise.columns)))
        axes[1].set_yticks(range(len(pivot_noise.index)))
        axes[1].set_xticklabels(pivot_noise.columns)
        axes[1].set_yticklabels(pivot_noise.index)
        axes[1].set_xlabel('min_samples')
        axes[1].set_ylabel('eps')
        axes[1].set_title('Noise Ratio', fontweight='bold')
        plt.colorbar(im2, ax=axes[1])
        
        plt.tight_layout()
        plt.show()
    
    return {
        'results_df': results_df,
        'best_params': best_params.to_dict(),
        'best_eps': best_params['eps'],
        'best_min_samples': int(best_params['min_samples'])
    }


def tune_gmm(X, n_components_range=range(2, 11), 
             covariance_types=['full', 'tied', 'diag', 'spherical'],
             random_state=42, plot=True):
    """
    Tune GMM hyperparameters using BIC and AIC scores.
    
    Args:
        X: Feature matrix (must be dense for GMM)
        n_components_range: Range of component numbers to try
        covariance_types: List of covariance types to try
        random_state: Random seed
        plot: Whether to display visualization
        
    Returns:
        Dictionary with tuning results and best parameters
    """
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    # Convert sparse to dense if needed
    if hasattr(X, 'toarray'):
        print("Converting sparse matrix to dense for GMM...")
        X = X.toarray()
    
    print("\n" + "="*60)
    print("GMM Hyperparameter Tuning")
    print("="*60)
    
    results = []
    
    for n_components in n_components_range:
        for cov_type in covariance_types:
            try:
                gmm = GaussianMixture(n_components=n_components, 
                                     covariance_type=cov_type,
                                     random_state=random_state,
                                     n_init=10)
                gmm.fit(X)
                
                bic = gmm.bic(X)
                aic = gmm.aic(X)
                
                # Try to compute silhouette if possible
                try:
                    labels = gmm.predict(X)
                    sil_score = silhouette_score(X, labels)
                except:
                    sil_score = np.nan
                
                results.append({
                    'n_components': n_components,
                    'covariance_type': cov_type,
                    'bic': bic,
                    'aic': aic,
                    'silhouette': sil_score,
                    'converged': gmm.converged_
                })
                
                print(f"n_components={n_components:2d}, cov_type={cov_type:10s} | "
                      f"BIC: {bic:12.2f} | AIC: {aic:12.2f} | "
                      f"Silhouette: {sil_score:6.3f} | Converged: {gmm.converged_}")
                
            except Exception as e:
                print(f"n_components={n_components:2d}, cov_type={cov_type:10s} | Failed: {str(e)}")
    
    results_df = pd.DataFrame(results)
    
    # Find best parameters (lowest BIC is preferred for model selection)
    converged_results = results_df[results_df['converged'] == True]
    
    if len(converged_results) > 0:
        best_idx_bic = converged_results['bic'].idxmin()
        best_params_bic = results_df.loc[best_idx_bic]
        
        best_idx_aic = converged_results['aic'].idxmin()
        best_params_aic = results_df.loc[best_idx_aic]
        
        print("\n" + "="*60)
        print("Best Parameters (by BIC - recommended):")
        print(f"  n_components = {int(best_params_bic['n_components'])}")
        print(f"  covariance_type = {best_params_bic['covariance_type']}")
        print(f"  BIC = {best_params_bic['bic']:.2f}")
        print(f"  AIC = {best_params_bic['aic']:.2f}")
        
        print("\nBest Parameters (by AIC):")
        print(f"  n_components = {int(best_params_aic['n_components'])}")
        print(f"  covariance_type = {best_params_aic['covariance_type']}")
        print(f"  BIC = {best_params_aic['bic']:.2f}")
        print(f"  AIC = {best_params_aic['aic']:.2f}")
        print("="*60 + "\n")
        
        best_params = best_params_bic  # Use BIC as default
    else:
        print("\n  Warning: No converged models found. Try adjusting parameters.")
        best_params = results_df.iloc[0]
    
    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        for cov_type in covariance_types:
            subset = results_df[results_df['covariance_type'] == cov_type]
            axes[0].plot(subset['n_components'], subset['bic'], 
                        marker='o', label=cov_type)
            axes[1].plot(subset['n_components'], subset['aic'], 
                        marker='o', label=cov_type)
        
        axes[0].set_xlabel('Number of Components')
        axes[0].set_ylabel('BIC')
        axes[0].set_title('BIC Scores (Lower is Better)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('AIC')
        axes[1].set_title('AIC Scores (Lower is Better)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return {
        'results_df': results_df,
        'best_params': best_params.to_dict(),
        'best_n_components': int(best_params['n_components']),
        'best_covariance_type': best_params['covariance_type']
    }
