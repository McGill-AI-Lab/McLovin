import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from src.core.embedder import Embedder  # Replace with your embedder

load_dotenv(dotenv_path='.env')

def onboard_fake_profiles(csv_path: str, index_name: str = "matching-index"):
    # Read CSV
    df = pd.read_csv(csv_path)

    # Initialize Pinecone
    pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
    index = pc.Index(host="https://matching-index-goovj0m.svc.aped-4627-b74a.pinecone.io")

    # Initialize Embedder
    embedder = Embedder()

    # Process each profile
    for _, row in df.iterrows():
        # Generate embeddings for bio and preferences
        bio_embedding = embedder.process_profile(row['bio'])
        pref_embedding = embedder.process_profile(row['preferences'])

        # Upsert to Pinecone with metadata
        index.upsert(
            vectors=[
                {
                    "id": row['user_id'],
                    "values": bio_embedding,
                    "metadata": {"fake": True, "type": "bio"}
                },
                {
                    "id": f"{row['user_id']}_preferences",
                    "values": pref_embedding,
                    "metadata": {"fake": True, "type": "preferences"}
                }
            ],
            namespace="bio-embeddings"
        )

if __name__ == "__main__":
    onboard_fake_profiles("path/to/fake_profiles.csv")
