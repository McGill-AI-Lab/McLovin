import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv
from src.core.embedder import Embedder
from src.core.profile import UserProfile, Faculty, Grade, Ethnicity


load_dotenv(dotenv_path='.env')

class Embedder:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        '''
        Initializes the BioEmbedder with SBERT model.
        '''
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SentenceTransformer(model_name)
        self.index_name = 'matching-index'
        self.dimension = 384
        self.metric = 'cosine'
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
        This is a helper for process_profile below.

        Output: vector embedding (1D np array)
        """
        # Prepare the input
        embedding = self.model.encode(text, convert_to_numpy=True)

        return embedding  # Return as 1D array

    def process_profile(self, profile):
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

        # we want to upsert into pinecone 2 different records: bio and pref
        # make a general metadata info (only the type is different) for both records
        common_metadata = {
            'age': profile.age,  # int
            'grade': profile.grade.value,  # enum Grade
            'faculty': profile.faculty.value,  # enum faculty
            'ethnicity': [str(e.value) for e in profile.ethnicity],  # list of enum ethnicities
            'major': profile.major,  # List of Strings
            'bio': profile.bio,
            'preferences': profile.preferences,
            'cluster_': profile.cluster_id,
        }

        bio_record = {
            "id": profile.user_id,
            "values": bio_embedding,
            "metadata": {
                 **common_metadata,
                 'type_of_vector': 'bio'
            }
        }

        pref_record = {
            "id": profile.user_id,
            "values": pref_embedding,
            "metadata": {
                 **common_metadata,
                 'type_of_vector': 'preferences'
            }
        }

        # upsert the records in their respective namespaces
        index.upsert(
            vectors = [bio_record],
            namespace='bio-embeddings'
        )
        index.upsert(
            vectors = [pref_record],
            namespace='preferences-embeddings'
        )

        # return the embedded vectors
        return [bio_embedding, pref_embedding]

if "__main__" == __name__:
    embedder = Embedder()

    test_profile = UserProfile(
        user_id="302",
        name="williamkiemlafond",
        age=21,
        grade=Grade.U2,
        ethnicity=[Ethnicity.WHITE, Ethnicity.LATIN],
        faculty=Faculty.ARTS,
        major=["Design Engineering"],
        bio="I love coding  ",
        preferences= 'Likes korean looking girls with big bumboclats',
    )
    embedding = embedder.process_profile(test_profile) # this should save 2 records : one for bio and one for preferences
    assert len(embedding[0]) == 384 # for bio vector
    assert len(embedding[1]) == 384 # for pref vector
    print("Profile stored in Pinecone")
    print(test_profile.tostring())
