from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
#from src.core.profile import UserProfile
from pinecone import Pinecone
import os
#import time
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

def cluster_users(n_clusters: int, random_state: int):
    """
        Cluster user bios using K-means clustering.

        Args:
            n_clusters (int): Number of clusters to create
            random_state (int): Random seed for reproducibility

        Returns:
            Tuple containing:
            - List of cluster assignments for each user
            - Cluster centroids
            - Fitted KMeans model
        """

    # get a list of all embedded bios from pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
    index = pc.Index(host="https://matching-index-goovj0m.svc.aped-4627-b74a.pinecone.io")

    # once MongoDB is implemented, we can just use userID[]
    all_bio_records = index.query(
        vector=[float(0)] * 384,
        namespace='bio-embeddings',
        filter={},
        top_k=1000,
        include_values=True
    )

    # extract all bio embeddings and convert into 2D numpy array
    all_bio_embeddings = []
    user_ids = [] # a list of user ids to change their cluster_ids attributes
    for match in all_bio_records['matches']:
        all_bio_embeddings.append(match['values'])
        user_ids.append(match['id'])


    np_bio_embeddings = np.array(all_bio_embeddings)

    # normalize the distributions of each np_array.
    scaler = StandardScaler()
    normalized_bio_embeddings = scaler.fit_transform(np_bio_embeddings)

    # perform k-means clustering
    kmeans = KMeans(
        n_clusters=n_clusters, # to start, 10 clusters
        random_state=random_state,
        n_init='auto' # increase if data is more sparse than expected
    )

    # fit the kmeans vectorspace by finding the centroids of bios
    # AND predict each embeddings assigned centroids
    cluster_assignements = kmeans.fit_predict(normalized_bio_embeddings)

    # return tupple -> (assignements, centroids, model, user_ids, normalized_embd)
    return (cluster_assignements, kmeans.cluster_centers_, kmeans, user_ids, normalized_bio_embeddings)


# helper function to help debug or visualize the kmeans performed
def analyze_clusters(cluster_assignments, centroids, normalized_embeddings):
    cluster_stats = {}
    unique_clusters = np.unique(cluster_assignments)

    for cluster_idx in unique_clusters:
            # Get points in this cluster
            cluster_members = np.where(cluster_assignments == cluster_idx)[0]
            cluster_points = normalized_embeddings[cluster_members]

            # Calculate mean distance to centroid
            centroid = centroids[int(cluster_idx)]  # Convert to int for indexing
            distances = np.linalg.norm(cluster_points - centroid, axis=1)
            mean_distance = np.mean(distances)

            # Calculate density (can be defined as points per unit volume, using mean distance as radius)
            volume = np.pi * (mean_distance ** 3) * 4/3  # Assuming spherical clusters
            density = len(cluster_members) / volume if volume > 0 else 0

            cluster_stats[f'cluster_{cluster_idx}'] = {
                'size': len(cluster_members),
                'percentage': len(cluster_members) / len(cluster_assignments) * 100,
                'mean_distance': mean_distance,
                'density': density
            }

    return cluster_stats

#if "__main__" == __name__:
 #   print(cluster_users(10,101))
