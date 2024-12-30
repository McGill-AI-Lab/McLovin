from src.core.bio_embedder import BioEmbedder
from src.core.profile import UserProfile, Faculty, Grade, Ethnicity

def test_embeddings():
    embedder = BioEmbedder()

    test_profile = UserProfile(
        user_id="mybitch",
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
    assert len(embedding) == 384
    print("Profile stored in Pinecone")
    print(test_profile.tostring())
