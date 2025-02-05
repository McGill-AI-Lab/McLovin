from pinecone import Pinecone
from typing import Dict, Tuple
import os
from dotenv import load_dotenv
from cluster_users import cluster_users, analyze_clusters
from evaluation import evaluate_kmeans, find_optimal_k
import logging
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def setup_logging():
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join(output_dir, f'clustering_run_{timestamp}.log')

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return log_filename

def visualize_evaluation(results: Dict, optimal_k: int) -> None:
    """Plot evaluation metrics and mark optimal k."""
    output_dir = 'outputs'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # elbow curve
    axes[0].plot(results['k_values'], results['inertias'], 'bo-')
    axes[0].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
    axes[0].set_xlabel('Number of clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')

    # silhouette scores
    axes[1].plot(results['k_values'], results['silhouette_scores'], 'ro-')
    axes[1].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Number of clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')

    # Calinski-Harabasz scores
    axes[2].plot(results['k_values'], results['calinski_scores'], 'go-')
    axes[2].axvline(x=optimal_k, color='r', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Number of clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Index')

    # Add a title showing the optimal k
    fig.suptitle(f'Clustering Evaluation Metrics (Optimal k={optimal_k})', fontsize=12)

    plt.tight_layout()
    output_file = os.path.join(output_dir, f'kmeans_evaluation_{timestamp}.png')
    plt.savefig(output_file)
    plt.close()

    logging.info(f"Saved evaluation plots to {output_file}")

def update_cluster_assignments(index, user_ids, cluster_assignments):
    """
    Update Pinecone vectors with their assigned cluster IDs
    """
    logging.info("Updating cluster assignments in Pinecone...")
    batch_size = 100  # Process in batches to avoid overwhelming the API

    for i in range(0, len(user_ids), batch_size):
        batch_ids = user_ids[i:i + batch_size]
        batch_assignments = cluster_assignments[i:i + batch_size]

        for user_id, cluster_id in zip(batch_ids, batch_assignments):
            index.update(
                id=user_id,
                namespace='bio-embeddings',
                set_metadata={'cluster_id': int(cluster_id)}
            )

    logging.info("Cluster assignments update complete")

def get_users_in_cluster(index, cluster_id):
    """
    Retrieve all users in a specific cluster
    """
    results = index.query(
        vector=[float(0)] * 384,
        namespace='bio-embeddings',
        filter={"cluster_id": cluster_id},
        top_k=1000,
        include_metadata=True
    )
    return results['matches']

if __name__ == "__main__":
    # Set up logging
    log_file = setup_logging()
    logging.info("Starting clustering process")

    # Load environment variables and initialize Pinecone
    load_dotenv(dotenv_path='.env')
    logging.info("Initializing Pinecone connection")
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
    index = pc.Index(host="https://matching-index-goovj0m.svc.aped-4627-b74a.pinecone.io")

    # Get all embeddings
    logging.info("Retrieving all user embeddings...")
    all_bio_records = index.query(
        vector=[float(0)] * 384,
        namespace='bio-embeddings',
        filter={},
        top_k=1000,
        include_values=True
    )

    # Extract embeddings and user IDs
    embeddings = []
    user_ids = []
    for match in all_bio_records['matches']:
        embeddings.append(match['values'])
        user_ids.append(match['id'])

    embeddings = np.array(embeddings)

    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)

    # Evaluate to find optimal k
    logging.info("Evaluating optimal number of clusters...")
    results = evaluate_kmeans(embeddings_normalized)
    optimal_k = find_optimal_k(results)
    logging.info(f"Selected optimal number of clusters: {optimal_k}")

    # Visualize evaluation metrics with optimal k
    visualize_evaluation(results, optimal_k)

    # Perform clustering with optimal k
    logging.info(f"Performing final clustering with k={optimal_k}...")
    assignments, centroids, model, _, _ = cluster_users(optimal_k, 101)

    # Update Pinecone with new cluster assignments
    logging.info("Updating all users with new cluster assignments...")
    update_cluster_assignments(index, user_ids, assignments)

    # Analyze and log results
    stats = analyze_clusters(assignments, centroids, embeddings_normalized)
    logging.info("\nClustering Results:")
    for cluster, data in stats.items():
        logging.info(f"{cluster}:")
        logging.info(f"  Members: {data['size']}")
        logging.info(f"  Percentage: {data['percentage']:.2f}%")

    # Log users in each cluster
    logging.info("\nUsers in each cluster:")
    for cluster_id in range(optimal_k):  # Use optimal_k instead of len(stats)
        cluster_users = get_users_in_cluster(index, cluster_id)
        logging.info(f"\nCluster {cluster_id} users:")
        user_ids_in_cluster = [user['id'] for user in cluster_users]
        logging.info(f"  User IDs: {', '.join(user_ids_in_cluster)}")

    logging.info("Clustering process complete")
