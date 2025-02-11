# Pseudocode for fetching cluster data
def get_cluster_users():
    clusters = defaultdict(list)
    # Query Pinecone for all users with cluster_id metadata
    results = pinecone_index.query(
        vector=[0]*dimension,  # Dummy query
        top_k=10000,
        include_metadata=True,
        filter={"cluster_id": {"$exists": True}}
    )
    for match in results['matches']:
        cluster_id = match['metadata']['cluster_id']
        clusters[cluster_id].append(match['metadata'])
    return clusters

# Main execution
if __name__ == "__main__":
    clusters = get_cluster_users()
    matches = run_matching(clusters)
    print(f"Generated {len(matches)} optimal pairs")
