from pinecone import Pinecone
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
index = pc.Index("matching-index")

# Check bio embeddings
bio_stats = index.describe_index_stats(namespace="bio-embeddings")
print(f"Bio embeddings stats: {bio_stats}")

# Check preference embeddings
pref_stats = index.describe_index_stats(namespace="preferences-embeddings")
print(f"Preference embeddings stats: {pref_stats}")

# Try to fetch one record from each namespace
bio_sample = index.query(
    vector=[0.0] * 384,
    namespace="bio-embeddings",
    include_values=True,
    include_metadata=True,
    top_k=1
)
print("\nSample bio record:", bio_sample)

pref_sample = index.query(
    vector=[0.0] * 384,
    namespace="preferences-embeddings",
    include_values=True,
    include_metadata=True,
    top_k=1
)
print("\nSample preference record:", pref_sample)
