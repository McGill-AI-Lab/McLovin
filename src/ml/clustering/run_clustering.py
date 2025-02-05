from pinecone import Pinecone
import os
from dotenv import load_dotenv
from cluster_users import cluster_users, analyze_clusters
from evaluation import evaluate_kmeans, find_optimal_k
import logging
from datetime import datetime
import numpy as np
from sklearn.preprocessing import StandardScaler

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

    # Load environment variables
    load_dotenv(dotenv_path='.env')

    # Initialize Pinecone
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
