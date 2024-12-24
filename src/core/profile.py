from enum import Enum
from dataclasses import dataclass
from typing_extensions import List

class Faculty(Enum):
    MED_DENT = 0
    MANAGEMENT = 1
    SCIENCE = 2
    ARTS = 3
    ARTS_AND_SCIENCE = 4
    ENGINEERING = 5
    EDUCATION = 6
    LAW = 7


class Grade(Enum):
     U0 = 0
     U1 = 1
     U2 = 2
     U3 = 3
     U4_OR_MORE = 4
     MASTER = 5
     PHD = 6


class Ethnicity(Enum):
    WHITE = 0
    BLACK = 1
    LATIN = 2
    EAST_ASIAN = 3
    SOUTHEAST_ASIAN = 4
    SOUTH_ASIAN = 5
    MIDDLE_EASTERN = 6
    INDIGENOUS = 7


@dataclass
class UserProfile:
    user_id: str # not important for encoding
    name: str # not important for encoding
    age: int
    grade: Grade
    ethnicity: List[Ethnicity]
    faculty: Faculty
    major: List[str] # embedding vector
    bio: str # embedding vector

# Note: major and faculty might not be necessarily related (ex.SE - CS >> Civil Eng - SE)
# might need to leverage LLM or any algorithm with Cosine Similarity

if __name__ == "__main__":
    print(len(Faculty))
