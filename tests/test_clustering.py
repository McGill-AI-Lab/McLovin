from src.ml.clustering.cluster_users import cluster_users, analyze_clusters

def test_clustering():
    # Perform clustering
        assignments, centroids, model = cluster_users(10, 69)

        # Analyze results
        stats = analyze_clusters(assignments, centroids)

        # Print results
        print("\nClustering Results:")
        print("-" * 50)
        for cluster, data in stats.items():
            print(f"{cluster}:")
            print(f"  Members: {data['size']}")
            print(f"  Percentage: {data['percentage']:.2f}%")

        # Calculate and print overall clustering metrics
        inertia = model.inertia_
        print("\nClustering Metrics:")
        print(f"Total Inertia (within-cluster sum of squares): {inertia:.2f}")
