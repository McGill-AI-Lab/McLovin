from src.core.embedder import Embedder
from src.core.profile import UserProfile, Faculty, Grade, Ethnicity

def test_embeddings():
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
