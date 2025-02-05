from pinecone import Pinecone
import os
from dotenv import load_dotenv
from cluster_users import cluster_users, analyze_clusters
from evaluation import main as run_evaluation

def update_cluster_assignments(index, user_ids, cluster_assignments):
    """
    Update Pinecone vectors with their assigned cluster IDs
    """
    for user_id, cluster_id in zip(user_ids, cluster_assignments):
        # Update metadata for each vector
        index.update(
            id=user_id,
            namespace='bio-embeddings',
            set_metadata={'cluster_id': int(cluster_id)}
        )

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
    # Load environment variables
    load_dotenv(dotenv_path='.env')

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
    index = pc.Index(host="https://matching-index-goovj0m.svc.aped-4627-b74a.pinecone.io")

    # First, run evaluation to determine optimal number of clusters
    print("Running cluster evaluation...")
    run_evaluation()  # This runs your evaluation script

    # Then perform clustering with the optimal number of clusters
    print("\nPerforming final clustering...")
    assignments, centroids, model, user_ids, norm_embeddings = cluster_users(10, 101)  # You might want to use the optimal_k from evaluation

    # Update Pinecone with cluster assignments
    print("\nUpdating Pinecone with cluster assignments...")
    update_cluster_assignments(index, user_ids, assignments)

    # Analyze and print results
    stats = analyze_clusters(assignments, centroids, norm_embeddings)
    print("\nClustering Results:")
    for cluster, data in stats.items():
        print(f"{cluster}:")
        print(f"  Members: {data['size']}")
        print(f"  Percentage: {data['percentage']:.2f}%")

    # Optional: Print example of users in each cluster
    print("\nSample users from each cluster:")
    for cluster_id in range(len(stats)):
        cluster_users = get_users_in_cluster(index, cluster_id)
        print(f"\nCluster {cluster_id} sample users:")
        for user in cluster_users[:3]:  # Show first 3 users from each cluster
            print(f"  User ID: {user['id']}")
