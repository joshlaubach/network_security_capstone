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
      - Avoids O(n²) memory for silhouette_score calculation
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
            X: Training features (DataFrame with categorical columns as strings)
            y: Not used (unsupervised), included for API consistency
        """
        # Encode categorical features (OneHot for clustering)
        X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals_for_clustering(X)
        
        # Apply feature scaling
        X_scaled, self.scaler_obj = scale_features(X_encoded, method=self.scaler_method)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        n_samples = len(X_scaled)
            
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10
        )
        self.model.fit(X_scaled)
        
        # Compute silhouette score on subsample if dataset is large
        # (silhouette_score has O(n²) memory complexity)
        if n_samples > self.max_silhouette_samples:
            print(f"[INFO] Dataset has {n_samples:,} samples (>{self.max_silhouette_samples:,})")
            print(f"       Computing silhouette on {self.max_silhouette_samples:,} subsample")
            print(f"       (prevents {n_samples**2/1e9:.1f}B distance calculations → OOM)")
            
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
        
        print(f"[INFO] K-Means fitted with {self.n_clusters} clusters")
        print(f"[INFO] Silhouette Score: {self.silhouette_score_:.4f}")
        print(f"[INFO] Anomaly threshold set at {self.threshold:.4f}")
        print(f"[INFO] Categorical cols encoded: {len(self.categorical_cols)}")
        print(f"[INFO] Scaling: {self.scaler_method}, Distance: {self.distance_metric}")
        
        return self
    
    def _compute_distances(self, X):
        """
        Compute distance from each point to its assigned cluster center.
        Uses the specified distance metric.
        """
        labels = self.model.predict(X)
        centers = self.model.cluster_centers_
        
        distances = np.zeros(len(X))
        
        if self.distance_metric == 'euclidean':
            for i in range(len(X)):
                distances[i] = np.linalg.norm(X[i] - centers[labels[i]])
        elif self.distance_metric == 'manhattan':
            for i in range(len(X)):
                distances[i] = np.sum(np.abs(X[i] - centers[labels[i]]))
        elif self.distance_metric == 'cosine':
            for i in range(len(X)):
                dot_product = np.dot(X[i], centers[labels[i]])
                norm_product = np.linalg.norm(X[i]) * np.linalg.norm(centers[labels[i]])
                distances[i] = 1 - (dot_product / (norm_product + 1e-10))
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        
        return distances
    
    def predict(self, X):
        """
        Predict anomaly labels: 0 = normal, 1 = anomaly.
        
        Args:
            X: Features to predict on (with categorical columns as strings)
            
        Returns:
            Binary labels (0/1)
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        distances = self._compute_distances(X_scaled)
        predictions = (distances > self.threshold).astype(int)
        
        return predictions
    
    def decision_function(self, X):
        """
        Return anomaly scores (higher = more anomalous).
        
        Args:
            X: Features to score (with categorical columns as strings)
            
        Returns:
            Anomaly scores (distance from cluster center)
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
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
      - For large datasets (>100K), trains on a subsample to avoid O(n²) memory
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
            X: Training features (DataFrame with categorical columns as strings)
            y: Not used (unsupervised)
        """
        # Encode categorical features
        X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals_for_clustering(X)
        
        # Apply feature scaling
        X_scaled, self.scaler_obj = scale_features(X_encoded, method=self.scaler_method)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        n_samples = len(X_scaled)
        
        # Memory-efficient: subsample if dataset is too large
        if n_samples > self.max_train_samples:
            print(f"[INFO] Dataset has {n_samples:,} samples (>{self.max_train_samples:,})")
            print(f"       Subsampling to {self.max_train_samples:,} for DBSCAN training")
            print(f"       (prevents {n_samples**2/1e9:.1f}B distance calculations → OOM)")
            
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
        
        print(f"[INFO] DBSCAN fitted:")
        print(f"       Clusters found: {n_clusters}")
        print(f"       Noise points: {n_noise:,} ({100*n_noise/n_samples:.2f}%)")
        print(f"       Trained on: {len(X_train_subsample):,}/{n_samples:,} samples")
        print(f"       Categorical cols encoded: {len(self.categorical_cols)}")
        print(f"       Scaling: {self.scaler_method}, Distance: {self.metric}")
        
        return self
    
    def predict(self, X):
        """
        Predict anomaly labels for new data (memory-efficient).
        
        Strategy:
          - Use NearestNeighbors to find closest training sample
          - If nearest neighbor is noise or far away, mark as anomaly
        
        Args:
            X: Features to predict on (with categorical columns as strings)
            
        Returns:
            Binary labels (0 = normal, 1 = anomaly)
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        # Use NearestNeighbors instead of fit_predict (memory efficient)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(self.X_train_scaled)
        
        distances, indices = nn.kneighbors(X_scaled)
        
        # Get labels of nearest neighbors
        nearest_labels = self.train_labels[indices.flatten()]
        
        # Mark as anomaly if:
        # 1. Nearest neighbor is noise (-1), OR
        # 2. Distance to nearest neighbor > eps
        predictions = np.zeros(len(X_scaled), dtype=int)
        predictions[(nearest_labels == -1) | (distances.flatten() > self.eps)] = 1
        
        return predictions
    
    def decision_function(self, X):
        """
        Return anomaly scores based on distance to nearest cluster (memory-efficient).
        
        Returns:
            Scores where points far from clusters get high values
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        # Use NearestNeighbors for scoring (memory efficient)
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=1, metric=self.metric)
        nn.fit(self.X_train_scaled)
        
        distances, indices = nn.kneighbors(X_scaled)
        
        # Get labels of nearest neighbors
        nearest_labels = self.train_labels[indices.flatten()]
        
        # Score based on:
        # 1. Distance to nearest neighbor (normalized by eps)
        # 2. Whether nearest neighbor is noise
        scores = distances.flatten() / self.eps
        scores[nearest_labels == -1] = 1.0  # Max score if nearest is noise
        
        return scores
    
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
            X: Training features (DataFrame with categorical columns as strings)
            y: Not used (unsupervised)
        """
        # Encode categorical features
        X_encoded, self.encoder_obj, self.categorical_cols = encode_categoricals_for_clustering(X)
        
        # Apply feature scaling
        X_scaled, self.scaler_obj = scale_features(X_encoded, method=self.scaler_method)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
        
        n_samples = len(X_scaled)
        
        # Memory-efficient: subsample if dataset is too large
        # GMM with 'full' covariance computes n_features x n_features covariance matrices
        # for each component during EM iterations (memory intensive)
        if n_samples > self.max_train_samples:
            print(f"[INFO] Dataset has {n_samples:,} samples (>{self.max_train_samples:,})")
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
        # (score_samples is O(n) not O(n²), so safe to use full dataset)
        log_likelihoods = self.model.score_samples(X_scaled)
        
        # Set threshold at contamination percentile
        # Lower log-likelihood = more anomalous
        self.threshold = np.percentile(log_likelihoods, 100 * self.contamination)
        
        print(f"[INFO] GMM fitted with {self.n_components} components")
        print(f"[INFO] Trained on: {len(X_train_subsample):,}/{n_samples:,} samples")
        print(f"[INFO] Log-likelihood threshold: {self.threshold:.4f}")
        print(f"[INFO] Covariance type: {self.covariance_type}")
        print(f"[INFO] Categorical cols encoded: {len(self.categorical_cols)}")
        print(f"[INFO] Scaling: {self.scaler_method}")
        
        # Check convergence
        if self.model.converged_:
            print(f"[INFO] GMM converged in {self.model.n_iter_} iterations")
        else:
            print(f"[WARN] GMM did not converge!")
        
        return self
    
    def predict(self, X):
        """
        Predict anomaly labels: 0 = normal, 1 = anomaly.
        
        Args:
            X: Features to predict on (with categorical columns as strings)
            
        Returns:
            Binary labels
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
            X_scaled = X_scaled.values
            
        log_likelihoods = self.model.score_samples(X_scaled)
        
        # Points with log-likelihood below threshold are anomalies
        predictions = (log_likelihoods < self.threshold).astype(int)
        
        return predictions
    
    def decision_function(self, X):
        """
        Return anomaly scores (negative log-likelihood).
        Higher scores = more anomalous.
        
        Args:
            X: Features to score (with categorical columns as strings)
            
        Returns:
            Anomaly scores
        """
        # Encode categorical features
        X_encoded, _, _ = encode_categoricals_for_clustering(
            X, encoder=self.encoder_obj
        )
        
        # Apply same scaling as training
        X_scaled, _ = scale_features(X_encoded, method=self.scaler_method, scaler=self.scaler_obj)
        
        if isinstance(X_scaled, pd.DataFrame):
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


def get_predictions_all_models(models, X_test):
    """
    Get predictions from all trained models.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        
    Returns:
        Dictionary of predictions and scores
    """
    results = {}
    
    for name, model in models.items():
        results[name] = {
            'predictions': model.predict(X_test),
            'scores': model.decision_function(X_test)
        }
    
    return results


def analyze_sus_vs_evil(models, X_test, y_sus, y_evil):
    """
    Analyze how well each model distinguishes 'sus' vs 'evil' outliers.
    
    Args:
        models: Dictionary of trained models
        X_test: Test features
        y_sus: Binary labels for 'sus' (in-distribution outliers)
        y_evil: Binary labels for 'evil' (out-of-distribution outliers)
        
    Returns:
        DataFrame with detection rates for each model and outlier type
    """
    results = []
    
    for name, model in models.items():
        preds = model.predict(X_test)
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
