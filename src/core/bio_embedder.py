from src.core import bio_embedder
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from src.core.profile import UserProfile, Grade, Faculty, Ethnicity
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

class BioEmbedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        '''
        Initializes the BioEmbedder with SBERT model.
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name)
        self.index_name = 'matching-index'
        self.dimension = 384
        self.metric = 'cosine'
        #self.namespace = 'bio-namespace'
        self.pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))

        # if this is the first instance of the index, create it
        if not self.pc.Index(self.index_name):
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

    def embed_text(self, text):
        """
        Create embedding for a single bio using BERT

        Output: vector embedding (1D np array)
        """
        # Prepare the input
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding  # Return as 1D array

    def process_profile(self, profile: UserProfile):
        '''
        Processes profile into a pinecone record and adds it to the remote index (Pinecone Database).

        Input: profile with all metadata (UserProfile)
        Output: vector embedding of the bio (list)
        '''
        # upsert the vectors, first define the index
        index = self.pc.Index(self.index_name)

        # 2 numerical values per profile
        bio_embedding = self.embed_text(profile.bio)
        pref_embedding = self.embed_text(profile.preferences)

        # Wait for the index to be ready
        while not self.pc.describe_index(self.index_name).status['ready']:
            time.sleep(1)

        # order matters here
        embeddings = [bio_embedding, pref_embedding]
        types = ['bio', 'preferences']

        for type_index, field in enumerate(embeddings):
            #print("type of vector", types[type_index])
            #print("field", field)
            # prepare the record to upsert
            metadata = {
                'type_of_vector': types[type_index],
                'age': profile.age,  # int
                'grade': profile.grade.value,  # enum Grade
                'faculty': profile.faculty.value,  # enum faculty
                'ethnicity': [str(e.value) for e in profile.ethnicity],  # list of enum ethnicities
                'major': profile.major,  # List of Strings
                'bio': profile.bio,
                'preferences': profile.preferences,
            }
            # record in index has form : record = {id, value, original text}
            record = {
                "id": profile.user_id + '_' + str(type_index),
                "values": field,
                "metadata": metadata
            }
            # upsert the record in pinecone
            index.upsert(
                vectors = [record],
                namespace='profile-embeddings'
            )

        # return the embedded vectors
        return embeddings
