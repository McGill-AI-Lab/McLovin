from src.core.embedder import Embedder
from src.core.profile import UserProfile, Faculty, Grade, Ethnicity

def test_embeddings():
    embedder = Embedder()

    test_profile = UserProfile(
        user_id="455",
        name="davidkhanhlafond",
        age=25,
        grade=Grade.U3,
        ethnicity=[Ethnicity.WHITE, Ethnicity.LATIN],
        faculty=Faculty.ARTS,
        major=["Design Engineering"],
        bio="I am a big dick wasian designer. I love gaming and watching k-dramas.",
        preferences= 'Likes korean looking girls with big bumboclats',
        cluster_id = 1
    )
    embedding = embedder.process_profile(test_profile) # this should save 2 records : one for bio and one for preferences
    assert len(embedding[0]) == 384 # for bio vector
    assert len(embedding[1]) == 384 # for pref vector
    print("Profile stored in Pinecone")
    print(test_profile.tostring())
