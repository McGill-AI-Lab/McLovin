from src.ml.clustering.evaluation import evaluate_kmeans, analyze_clusters
import numpy as np
from sklearn.cluster import KMeans

def test_kmeans_evaluation():
    # Create mock embeddings
    mock_embeddings = np.random.randn(20, 384)

    # Test evaluation
    results = evaluate_kmeans(mock_embeddings, k_range=range(2, 5))
    assert len(results['k_values']) == 3
    assert all(s >= -1 and s <= 1 for s in results['silhouette_scores'])

    # Test clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(mock_embeddings)
    stats = analyze_clusters(mock_embeddings, labels)
    assert len(stats) == 3
