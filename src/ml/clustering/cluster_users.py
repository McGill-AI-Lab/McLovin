from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.core.profile import UserProfile
from pinecone import Pinecone
import os
import time
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
    for match in all_bio_records['matches']:
        all_bio_embeddings.append(match['values'])
    np_bio_embeddings = np.array(all_bio_embeddings)

    # normalize the distributions of each np_array
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

    # return tupple -> (assignements, centroids, model)
    return (cluster_assignements, kmeans.cluster_centers_, kmeans)


# helper function to help debug or visualize the kmeans performed
def analyze_clusters(cluster_assignements, centroids):
    cluster_stats = {}
    unique_clusters = np.unique(cluster_assignements)

    for cluster in unique_clusters:
        cluster_members = np.where(cluster_assignements == cluster)[0]
        cluster_stats[f'cluster_{cluster}'] = {
            'size': len(cluster_members),
            'percentage': len(cluster_members) / len(cluster_assignements) * 100
        }

    return cluster_stats


if "__main__" == __name__:
    print(cluster_users(10,101))
