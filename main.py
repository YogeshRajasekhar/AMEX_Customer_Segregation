"""
Runs the full Amex customer-segmentation + contextual-bandit pipeline
end to end: feature engineering, the K-Means vs. DBSCAN comparison (with
the real DBSCAN mega-cluster finding), the business-chosen k=4 clustering,
persona descriptions, and the LinUCB bandit (with the real
wrong-then-fixed convergence story for Segment 2).
"""

import os

from src.bandit import (
    ACTIONS,
    CORRECTED_ALPHA,
    CORRECTED_N_ROUNDS,
    TRUE_BEST,
    compute_segment_stats,
    lift_over_random,
    run_bandit,
)
from src.clustering import (
    best_kmeans_k,
    business_kmeans,
    cluster_profile,
    cluster_sizes,
    dbscan_param_search,
    kmeans_silhouette_search,
    reduce_dimensions,
    scale_features,
)
from src.features import build_features
from src.personas import describe_personas

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "CC_GENERAL.csv")


def main():
    print("=" * 70)
    print("STEP 1: Feature engineering")
    print("=" * 70)
    df, features = build_features(DATA_PATH)
    print(f"Raw data shape: {df.shape}")
    print(f"Engineered feature set shape: {features.shape}")

    print()
    print("=" * 70)
    print("STEP 2: Scaling, PCA, and the honest K-Means vs. DBSCAN comparison")
    print("=" * 70)
    X_scaled, _ = scale_features(features)
    X_pca, _, cumvar = reduce_dimensions(X_scaled)
    print(f"PCA: {X_scaled.shape[1]} features reduced to {X_pca.shape[1]} components "
          f"(>=85% variance: {cumvar[X_pca.shape[1] - 1]:.3f})")

    kmeans_results = kmeans_silhouette_search(X_pca)
    optimal_k = best_kmeans_k(kmeans_results)
    print(f"K-Means silhouette-optimal k: {optimal_k} "
          f"(score={kmeans_results[optimal_k]['silhouette']:.4f})")

    dbscan_result = dbscan_param_search(X_pca)
    eps, min_samples = dbscan_result["best_params"]
    mega_cluster = max(
        (dbscan_result["best_labels"] == label).sum()
        for label in set(dbscan_result["best_labels"])
        if label != -1
    )
    print(f"DBSCAN silhouette-optimal config: eps={eps}, min_samples={min_samples} "
          f"(score={dbscan_result['best_score']:.4f})")
    print(f"  REAL FINDING: this collapses {mega_cluster} of {len(dbscan_result['best_labels'])} "
          f"customers ({100 * mega_cluster / len(dbscan_result['best_labels']):.1f}%) into ONE "
          f"mega-cluster -- higher-scoring than K-Means, but practically useless for personas.")

    print()
    print("=" * 70)
    print("STEP 3: Business-usable clustering (k=4, not silhouette-optimal)")
    print("=" * 70)
    labels, sil = business_kmeans(X_pca)
    features = features.copy()
    features["cluster"] = labels
    sizes = cluster_sizes(features)
    print(f"K-Means at k=4: silhouette={sil:.4f} (vs {kmeans_results[optimal_k]['silhouette']:.4f} "
          f"at the statistically 'optimal' k={optimal_k})")
    print("Segment sizes:")
    for cluster_id, n in sizes.items():
        print(f"  Segment {cluster_id}: {n} customers ({100 * n / len(features):.1f}%)")

    print()
    print("=" * 70)
    print("STEP 4: Personas, derived from the real per-segment statistics")
    print("=" * 70)
    profile = cluster_profile(features)
    personas = describe_personas(profile, sizes)
    for persona in personas.values():
        print(persona["description"])

    print()
    print("=" * 70)
    print("STEP 5: LinUCB contextual bandit (simulated reward, real segments)")
    print("=" * 70)
    segment_stats = compute_segment_stats(features)
    customer_segments = features["cluster"].values
    result = run_bandit(
        customer_segments, segment_stats, n_rounds=CORRECTED_N_ROUNDS, alpha=CORRECTED_ALPHA, seed=42
    )
    lift = lift_over_random(result["bandit_rewards"], result["random_rewards"])
    print(f"Random baseline conversion rate: {result['random_rewards'].mean():.4f}")
    print(f"LinUCB bandit conversion rate:   {result['bandit_rewards'].mean():.4f}")
    print(f"Lift over random: {lift:.1f}%")

    print()
    print("Per-segment learned-vs-true-optimal offer:")
    all_correct = True
    for segment in range(4):
        learned = result["learned_actions"][segment]
        true_action = TRUE_BEST[segment]
        correct = learned == true_action
        all_correct = all_correct and correct
        status = "CORRECT" if correct else f"WRONG (true best: {true_action})"
        print(f"  Segment {segment}: learned={learned}  [{status}]")

    verdict = "CONVERGED TO TRUE OPTIMUM for all segments" if all_correct else "DID NOT fully converge"
    print(f"\nVerification: bandit {verdict}")
    print(f"(Available actions: {', '.join(ACTIONS)})")


if __name__ == "__main__":
    main()
