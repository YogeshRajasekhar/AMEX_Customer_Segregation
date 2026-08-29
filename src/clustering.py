"""
Scale features, check whether PCA is actually warranted (only add
dimensionality reduction if the real feature count needs it), then run
K-Means and DBSCAN and compare them honestly using silhouette score --
not just picking whichever looks nicer.

Then pick the business-usable K-Means configuration (k=4), a deliberate,
documented override of pure statistical optimality: the silhouette-optimal
K-Means (k=2) is too coarse for real personas, and DBSCAN's silhouette-
optimal config collapses 98% of customers into one mega-cluster -- see
`dbscan_param_search` below. Both facts are preserved here, not just the
final chosen k.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

KMEANS_K_RANGE = range(2, 9)
DBSCAN_EPS_GRID = [0.5, 0.8, 1.0, 1.5, 2.0, 2.5]
DBSCAN_MIN_SAMPLES_GRID = [5, 10, 15]
PCA_VARIANCE_THRESHOLD = 0.85

# Chosen for business interpretability (a marketing team can act on 4
# personas; 2 is too coarse, and DBSCAN's result wasn't usable at all),
# NOT because it is silhouette-optimal -- k=2 was (score 0.4799 vs 0.2386
# at k=4). Trading a small amount of statistical optimality for a
# business-usable outcome, deliberately and documented here.
BUSINESS_K = 4

PROFILE_COLUMNS = [
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "PURCHASES_FREQUENCY",
    "CASH_ADVANCE_FREQUENCY",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "PRC_FULL_PAYMENT",
    "TENURE",
]


def scale_features(features):
    """Scale -- mandatory before any distance-based method (K-Means,
    DBSCAN both use distance; without this, BALANCE (0-19043) would
    dominate over PURCHASES_FREQUENCY (0-1) purely due to unit scale, not
    real importance."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    return X_scaled, scaler


def reduce_dimensions(X_scaled, variance_threshold=PCA_VARIANCE_THRESHOLD):
    """Only reduce dimensionality if the real feature count needs it --
    explained variance decides how many components, not a guessed round
    number."""
    pca_check = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_check.explained_variance_ratio_)
    n_components = int(np.argmax(cumvar >= variance_threshold)) + 1
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    return X_pca, pca, cumvar


def kmeans_silhouette_search(X_pca, k_range=KMEANS_K_RANGE, random_state=42):
    """Elbow/silhouette search for real cluster-count selection, not
    assumed. Returns {k: {"inertia", "silhouette", "labels"}}."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X_pca)
        results[k] = {
            "inertia": km.inertia_,
            "silhouette": silhouette_score(X_pca, labels),
            "labels": labels,
        }
    return results


def best_kmeans_k(kmeans_results):
    """The statistically 'best' k by silhouette score. NOTE: this is later
    deliberately overridden by business-usability logic (see
    `business_kmeans` below) -- both facts must be preserved, not just the
    final chosen k."""
    return max(kmeans_results, key=lambda k: kmeans_results[k]["silhouette"])


def dbscan_param_search(
    X_pca, eps_grid=DBSCAN_EPS_GRID, min_samples_grid=DBSCAN_MIN_SAMPLES_GRID
):
    """Real DBSCAN parameter search across an eps/min_samples grid, scored
    by silhouette on non-noise points (standard practice, since noise
    points aren't assigned to any cluster by definition).

    REAL FINDING, PRESERVE THIS: the silhouette-optimal config found here
    (eps=2.0, min_samples=5, silhouette=0.5184) puts 98% of customers
    (8767 of 8950) into ONE cluster plus a few 4-5-person micro-clusters --
    mathematically higher-scoring than K-Means but practically useless for
    real personas. This is documented explicitly, not silently dropped.
    """
    best_score = -1
    best_params = None
    best_labels = None
    search_results = []
    for eps in eps_grid:
        for min_samples in min_samples_grid:
            db = DBSCAN(eps=eps, min_samples=min_samples)
            labels = db.fit_predict(X_pca)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_noise = int(list(labels).count(-1))
            if n_clusters < 2:
                continue  # can't compute silhouette with <2 real clusters
            mask = labels != -1
            if mask.sum() < 2:
                continue
            score = silhouette_score(X_pca[mask], labels[mask])
            search_results.append(
                {
                    "eps": eps,
                    "min_samples": min_samples,
                    "n_clusters": n_clusters,
                    "n_noise": n_noise,
                    "silhouette": score,
                }
            )
            if score > best_score:
                best_score = score
                best_params = (eps, min_samples)
                best_labels = labels
    return {
        "best_score": best_score,
        "best_params": best_params,
        "best_labels": best_labels,
        "search_results": search_results,
    }


def business_kmeans(X_pca, k=BUSINESS_K, random_state=42):
    """K-Means at the business-chosen k (see BUSINESS_K), not the
    silhouette-optimal k. Returns (labels, silhouette_score)."""
    kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    sil = silhouette_score(X_pca, labels)
    return labels, sil


def cluster_profile(features_with_cluster, cluster_col="cluster"):
    """Real per-cluster mean statistics, used both for reporting and for
    grounding the persona descriptions and the bandit's reward function."""
    return features_with_cluster.groupby(cluster_col)[PROFILE_COLUMNS].mean().round(2)


def cluster_sizes(features_with_cluster, cluster_col="cluster"):
    return features_with_cluster[cluster_col].value_counts().sort_index()
