import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
#from ml.vectorstore.pinecone_client import PineconeClient
from core.profile import UserProfile, Grade, Faculty, Ethnicity

class BioEmbedder:
    def __init__(self, model_name='bert-base-uncased'):
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
        metadata = {
            'age': profile.age, # int
            'grade': profile.grade.value, # enum Grade
            'faculty': profile.faculty.value, # enum faculty
            'ethnicity': [e.value for e in profile.ethnicity], # list of enum ethnicities
            'major': profile.major # List of Strings
        }

        # store the profile data in one vector embedding in pinecone
        #self.pinecone.store_profile_embeddings(
        #    user_id=profile.user_id,
        #    bio_embedding=bio_embedding.tolist(), # turns np array into classic python list
        #    meta_data=metadata,
        #)

        # return the embedded bio
        return bio_embedding



# Usage example:
"""
embedder = BioEmbedder()

# Process single bio
bio = "Computer Science student interested in AI and machine learning..."
embedding = embedder.embed_bio(bio)

# Store in Weaviate
embedder.update_embeddings(weaviate_client, "user123", bio)
"""

if __name__ == "__main__":
    embedder = BioEmbedder()

    test_profile = UserProfile(
        user_id="test123",
        name="Test User",
        age=20,
        grade=Grade.U2,
        ethnicity=[Ethnicity.WHITE],
        faculty=Faculty.ENGINEERING,
        major=["Software Engineering"],
        bio="Computer Science student interested in AI and machine learning"
    )

    embedding = embedder.process_profile(test_profile)
    print(embedding)
    print(f"Embedding shape: {embedding.shape}")
    print("Profile stored in Pinecone")
