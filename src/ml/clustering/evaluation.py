from sklearn.cluster import KMeans
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from typing import Dict, Tuple
import seaborn as sns
from sklearn.model_selection import train_test_split
from pinecone import Pinecone
import os
from dotenv import load_dotenv

def evaluate_kmeans(embeddings: np.ndarray, k_range: range = range(2, 11)) -> Dict:
    """
    Evaluate KMeans clustering for different values of k.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        k_range: Range of k values to try
    """
    results = {
        'k_values': list(k_range),
        'inertias': [],
        'silhouette_scores': [],
        'calinski_scores': []
    }

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)

        results['inertias'].append(kmeans.inertia_)
        results['silhouette_scores'].append(silhouette_score(embeddings, labels))
        results['calinski_scores'].append(calinski_harabasz_score(embeddings, labels))

    return results

def visualize_evaluation(results: Dict) -> None:
    """Plot evaluation metrics."""

    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # elbow curve
    axes[0].plot(results['k_values'], results['inertias'], 'bo-')
    axes[0].set_xlabel('Number of clusters (k)')
    axes[0].set_ylabel('Inertia')
    axes[0].set_title('Elbow Method')

    # silhouette scores
    axes[1].plot(results['k_values'], results['silhouette_scores'], 'ro-')
    axes[1].set_xlabel('Number of clusters (k)')
    axes[1].set_ylabel('Silhouette Score')
    axes[1].set_title('Silhouette Analysis')

    # Calinski-Harabasz scores
    axes[2].plot(results['k_values'], results['calinski_scores'], 'go-')
    axes[2].set_xlabel('Number of clusters (k)')
    axes[2].set_ylabel('Calinski-Harabasz Score')
    axes[2].set_title('Calinski-Harabasz Index')

    plt.tight_layout()
    output_file = os.path.join(output_dir, 'kmeans_evaluation.png')
    plt.savefig(output_file)
    plt.close()

def analyze_clusters(embeddings: np.ndarray, labels: np.ndarray) -> Dict:
    """
    Analyze characteristics of each cluster.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        labels: Array of cluster assignments
    """
    cluster_stats = {}

    for cluster_id in np.unique(labels):
        cluster_mask = labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]

        cluster_stats[f'cluster_{cluster_id}'] = {
            'size': np.sum(cluster_mask),
            'percentage': np.mean(cluster_mask) * 100,
            'mean_distance_to_center': np.mean(np.linalg.norm(
                cluster_embeddings - np.mean(cluster_embeddings, axis=0), axis=1
            )),
            'density': np.mean([
                np.sum(np.linalg.norm(x - cluster_embeddings, axis=1) < 1)
                for x in cluster_embeddings
            ])
        }

    return cluster_stats

def main():
    # Load data from Pinecone
    load_dotenv(dotenv_path='.env')
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
    index = pc.Index(host="https://matching-index-goovj0m.svc.aped-4627-b74a.pinecone.io")

    all_bio_records = index.query(
        vector=[float(0)] * 384,
        namespace='bio-embeddings',
        filter={},
        top_k=1000,
        include_values=True
    )

    # Convert to numpy array
    embeddings = np.array([match['values'] for match in all_bio_records['matches']])
    print("embedding size: ", embeddings.shape)
    # Normalize embeddings
    scaler = StandardScaler()
    embeddings_normalized = scaler.fit_transform(embeddings)

    # Split data for validation
    train_embeddings, val_embeddings = train_test_split(
        embeddings_normalized, test_size=0.2, random_state=42
    )

    # Evaluate different k values
    print("Evaluating different numbers of clusters...")
    results = evaluate_kmeans(train_embeddings)
    visualize_evaluation(results)

    # Find optimal k using silhouette score
    optimal_k = results['k_values'][np.argmax(results['silhouette_scores'])]
    print(f"\nOptimal number of clusters based on silhouette score: {optimal_k}")

    # Final clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42,n_init='auto')
    train_labels = kmeans.fit_predict(train_embeddings)
    val_labels = kmeans.predict(val_embeddings)

    # Analyze clusters
    train_stats = analyze_clusters(train_embeddings, train_labels)
    val_stats = analyze_clusters(val_embeddings, val_labels)
    print("\n------------------------------------------------------------------")
    print("\nTraining Set Cluster Analysis:")
    for cluster_id, stats in train_stats.items():
        print(f"\n{cluster_id}:")
        print(f"Size: {stats['size']} ({stats['percentage']:.1f}%)")
        print(f"Mean distance to center: {stats['mean_distance_to_center']:.3f}")
        print(f"Density: {stats['density']:.3f}")

    print("\n------------------------------------------------------------------")
    print("\nValidation Set Cluster Analysis:")
    for cluster_id, stats in val_stats.items():
        print(f"\n{cluster_id}:")
        print(f"Size: {stats['size']} ({stats['percentage']:.1f}%)")
        print(f"Mean distance to center: {stats['mean_distance_to_center']:.3f}")
        print(f"Density: {stats['density']:.3f}")

if __name__ == "__main__":
    main()
