from core.bio_embedder import BioEmbedder
from core.profile import UserProfile, Faculty, Grade, Ethnicity

def test_embeddings():
    embedder = BioEmbedder()

    test_profile = UserProfile(
        user_id="oliver",
        name="Test User",
        age=20,
        grade=Grade.U0,
        ethnicity=[Ethnicity.WHITE],
        faculty=Faculty.ENGINEERING,
        major=["Software Engineering"],
        bio="Does not HATE cats and dogs",
        preferences= ''
    )

    embedding = embedder.process_profile(test_profile)
    print(f"Embedding shape: {len(embedding)}")
    print("Profile stored in Pinecone")

if __name__ == "__main__":
    test_embeddings()
