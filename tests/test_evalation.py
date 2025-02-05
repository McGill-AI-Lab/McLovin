from src.ml.clustering.evaluation import evaluate_kmeans, analyze_clusters, main
import numpy as np
from sklearn.cluster import KMeans
import pytest
import os

def test_kmeans_evaluation():
    # Create mock embeddings
    mock_embeddings = np.random.randn(20, 384)

    # Test evaluation
    results = evaluate_kmeans(mock_embeddings, k_range=range(2, 5))

    # Basic assertions
    assert len(results['k_values']) == 3  # because range(2,5) has 3 values
    assert all(s >= -1 and s <= 1 for s in results['silhouette_scores'])
    assert all(isinstance(x, (int, float)) for x in results['inertias'])
    assert all(isinstance(x, (int, float)) for x in results['calinski_scores'])

    # Test clustering with explicit number of clusters
    kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto')
    kmeans.fit(mock_embeddings)
    labels = kmeans.labels_
    stats = analyze_clusters(labels, kmeans.cluster_centers_)  # Changed argument order

    # Verify cluster statistics
    assert isinstance(stats, dict)
    for cluster_id, cluster_stats in stats.items():
        assert isinstance(cluster_stats, dict)
        assert 'size' in cluster_stats
        assert 'percentage' in cluster_stats


def test_output_directory():
    """Test if output directory is created and accessible"""
    output_dir = 'outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    assert os.path.exists(output_dir)
    assert os.access(output_dir, os.W_OK)

@pytest.mark.integration
def test_main_function():
    """Test the main evaluation function"""
    try:
        main()
        # Check if evaluation plot was created
        assert os.path.exists('outputs/kmeans_evaluation.png')
    except Exception as e:
        pytest.fail(f"Main evaluation function failed: {str(e)}")
