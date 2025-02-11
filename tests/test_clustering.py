from src.ml.clustering.cluster_users import cluster_users, analyze_clusters
import pytest
from pinecone import Pinecone
import os
from dotenv import load_dotenv

def test_clustering():
    assignments, centroids, model, user_ids = cluster_users(10, 69)

    # Verify we got user_ids
    assert len(user_ids) > 0
    assert len(assignments) == len(user_ids)

    # Analyze results
    stats = analyze_clusters(assignments, centroids)

    # Basic assertions
    assert len(stats) > 0

    # Check model properties
    if hasattr(model, 'inertia_'):
        assert model.inertia_ >= 0  # Only check if inertia_ exists

    assert hasattr(model, 'n_clusters')
    assert model.n_clusters == 10

def test_pinecone_connection():
    """Test if we can connect to Pinecone"""
    load_dotenv(dotenv_path='.env')
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
        index = pc.Index(host="https://matching-index-goovj0m.svc.aped-4627-b74a.pinecone.io")
        # Try a simple query
        response = index.query(
            vector=[float(0)] * 384,
            namespace='bio-embeddings',
            top_k=1
        )
        assert 'matches' in response
    except Exception as e:
        pytest.fail(f"Failed to connect to Pinecone: {str(e)}")
