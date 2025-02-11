import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import os
import time
from dotenv import load_dotenv
#from src.core.embedder import Embedder
from src.core.profile import UserProfile, Faculty, Grade, Gender, Ethnicity, SexualOrientationProbabilities


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
        print("DEBUG GRADE",profile.grade)

        common_metadata = {
            'age': profile.age,
            'grade': profile.grade.value,
            'faculty': profile.faculty.value,
            'ethnicity': [str(e.value) for e in profile.ethnicity],
            'major': profile.major,
            'gender': profile.gender.value,
            'bio': profile.bio,
            'preferences': profile.preferences,
            'cluster_id': profile.cluster_id,
            'fake': profile.fake,
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
    # for test, we can put fakeness = False, but might change

    fake = False # for now, we say our example is a real person
    current_gender = Gender.randomGender()
    # generate a random list for GOIS
    rand_genders = SexualOrientationProbabilities.generate_genders_of_interest(current_gender)

    #
    test_profile = UserProfile(
        user_id="user_yukimi",
        name="yukimi",
        age=22,
        gender=Gender.female,
        genders_of_interest=[Gender.male],
        grade=Grade.U3,
        ethnicity=[Ethnicity.EAST_ASIAN],
        faculty=Faculty.MANAGEMENT,
        major=["Accounting"],
        bio="I am japanese girl who likes to send stupid memes, but they are very funny. I like to draw and go out with friends. I am very cute.",
        preferences= 'I love boys who are asian, and even more wasian boys. Nerds in CS or in math programs are even cooler. I love my nerds!',
        fake=fake # default is True for synthetic, but we want a real person for this test

    )
    embedding = embedder.process_profile(test_profile) # this should save 2 records : one for bio and one for preferences
    assert len(embedding[0]) == 384 # for bio vector
    assert len(embedding[1]) == 384 # for pref vector
    print("Profile stored in Pinecone")
    print(test_profile.tostring())
