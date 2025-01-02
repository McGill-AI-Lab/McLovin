from sklearn.cluster import KMeans
import numpy as np
from src.core.profile import UserProfile
from pinecone import Pinecone
import os
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

def cluster_users():
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
    all_bio_embeddings = []

    for match in all_bio_records['matches']:
        all_bio_embeddings.append(match['values'])



if "__main__" == __name__:
    cluster_users()
