import pytest
from src.core.profile import UserProfile, Grade, Ethnicity, Faculty
from src.core.embedder import Embedder  # Replace with the actual module name

def generate_mock_profiles():
    """Generate 20 diverse mock profiles."""
    return [
        UserProfile(
            user_id=str(i + 1),
            name=f"user{i+1}",
            age=20 + (i % 5),
            grade=[Grade.U1, Grade.U2, Grade.U3, Grade.MASTER][i % 4],
            ethnicity=[Ethnicity.WHITE, Ethnicity.EAST_ASIAN, Ethnicity.BLACK, Ethnicity.LATIN][:(i % 4) + 1],
            faculty=[Faculty.ARTS, Faculty.SCIENCE, Faculty.ENGINEERING, Faculty.MANAGEMENT][i % 4],
            major=["Computer Science", "Economics", "Physics", "Design Engineering", "Biology"][(i % 5):],
            bio=f"This is a sample bio for user {i+1}. I like hobby {i % 3}",
            preferences=f"Looking for someone who enjoys interest {i % 3}",
        )
        for i in range(20)
    ]

@pytest.mark.parametrize("profile", generate_mock_profiles())
def test_embeddings(profile):
    """Test embeddings for each mock profile."""
    embedder = Embedder()

    # Process profile
    embedding = embedder.process_profile(profile)

    # Assert embeddings have the correct dimension
    assert len(embedding[0]) == 384  # Bio vector
    assert len(embedding[1]) == 384  # Preferences vector

    print(f"Profile stored in Pinecone for user {profile.user_id}")
    print(profile)
