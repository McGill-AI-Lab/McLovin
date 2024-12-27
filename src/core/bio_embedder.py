import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
#from ml.vectorstore.pinecone_client import PineconeClient
from core.profile import UserProfile, Grade, Faculty, Ethnicity
from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

class BioEmbedder:
    def __init__(self, model_name='google-bert/bert-base-uncased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        #self.pinecone = PineconeClient()

    # helper function for process_profile
    def embed_bio(self, bio: str) -> np.ndarray:
        """
        Create embedding for a single bio using BERT
        """
        # Prepare the input
        inputs = self.tokenizer(
            bio,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Get BERT embedding
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use [CLS] token embedding as bio representation
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding[0]  # Return as 1D array


    def process_profile(self, profile: UserProfile):
        # numerical value for profile
        bio_embedding = self.embed_bio(profile.bio)

        # store the profile data in one vector embedding in pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

        index_name='matching-index' # this index has 3 namespaces for now
        # if this is the first instance of the index, create it
        if not pc.has_index(index_name):
            pc.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        # Wait for the index to be ready
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)

        # upsert the vectors, first define the index
        index = pc.Index(index_name)

        # prepare the record to upsert
        metadata = {
            'type_of_vector': 'bio',
            'age': profile.age,  # int
            'grade': profile.grade.value,  # enum Grade
            'faculty': profile.faculty.value,  # enum faculty
            'ethnicity': [str(e.value) for e in profile.ethnicity],  # list of enum ethnicities
            'major': profile.major,  # List of Strings
            'bio': profile.bio
        }
        # record in index has form : record = {id, value, original text}
        record = {
            "id": profile.user_id,
            "values": bio_embedding.tolist(),
            "metadata": metadata
        }
        records = [record]

        # upsert the record into index
        index.upsert(
            vectors=records, # update/insert (upsert) the new record
            namespace="bios-namespace"
        )
        print("just upserted records:", records, "in index", index)

        # debug
        time.sleep(10)  # Wait for the upserted vectors to be indexed
        print("index stats after 10 sec", index.describe_index_stats())

        # return the embedded bio
        return bio_embedding


if __name__ == "__main__":
    embedder = BioEmbedder()

    test_profile = UserProfile(
        user_id="test123",
        name="Test User",
        age=20,
        grade=Grade.U2,
        ethnicity=[Ethnicity.WHITE, Ethnicity.EAST_ASIAN],
        faculty=Faculty.ENGINEERING,
        major=["Software Engineering"],
        bio="Computer Science student interested in AI and machine learning. Has fun doing random shit"
    )

    embedding = embedder.process_profile(test_profile)
    print(embedding)
    print(f"Embedding shape: {embedding.shape}")
    print("Profile stored in Pinecone")
