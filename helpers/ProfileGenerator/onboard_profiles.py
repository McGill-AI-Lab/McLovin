import pandas as pd
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from src.core.embedder import Embedder

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
    for i, row in df.iterrows():
        # Generate embeddings for bio and preferences
        #print("bio is:", row['bios'])
        bio_embedding = embedder.embed_text(row['bios'])
        pref_embedding = embedder.embed_text('')

        # create a common metadata
        common_metadata = {
            'gender':row['gender'],
            'major': row['major'],
            'bio': row['bios'],
            'preferences': "none for now",
            'cluster_id': -1,
        }

        bio_record = {
            "id": f"user_{i}",
            "values": bio_embedding,
            "metadata": {
                **common_metadata,
                'type_of_vector': 'bio',
                'fake': True,
            }
        }

        pref_record = {
            "id": f"user_{i}",
            "values": pref_embedding,
            "metadata": {
                **common_metadata,
                'type_of_vector': 'preferences',
                'fake': True,
            }
        }

        # Upsert to Pinecone with metadata
        index.upsert(
            vectors=[bio_record],
            namespace="bio-embeddings"
        )
        index.upsert(
            vectors = [pref_record],
            namespace='preferences-embeddings'
        )

if __name__ == "__main__":
    onboard_fake_profiles("outputs/profiles.csv")
