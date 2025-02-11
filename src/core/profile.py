from enum import Enum
from dataclasses import dataclass
from typing_extensions import List
import random

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

class Gender(Enum):
    male = 0
    female = 1
    nonbinary = 2
    other = 3

    @staticmethod
    def randomGender():
        roll = random.random()
        if roll < 0.49:
            return Gender.male
        elif roll < 0.98:
            return Gender.female
        elif roll < 0.985:
            return Gender.nonbinary
        else:
            return Gender.other




class SexualOrientationProbabilities:
    STRAIGHT_PROB = 0.93  # 93% heterosexual
    BISEXUAL_PROB = 0.04  # 4% bisexual
    GAY_PROB = 0.03      # 3% homosexual

    @staticmethod
    def generate_genders_of_interest(own_gender) -> List[Gender]:

        roll = random.random()

        if roll < SexualOrientationProbabilities.STRAIGHT_PROB:
            # Straight
            if own_gender == Gender.male:
                return [Gender.female]
            elif own_gender == Gender.female:
                return [Gender.male]
            else:  # For non-binary/other, randomly choose one
                return random.choice([[Gender.male], [Gender.female]])

        elif roll < (SexualOrientationProbabilities.STRAIGHT_PROB + SexualOrientationProbabilities.BISEXUAL_PROB):
            # Bisexual - interested in male and female
            return [Gender.male, Gender.female]

        else:
            # Gay/Lesbian
            if own_gender == Gender.male:
                return [Gender.male]
            elif own_gender == Gender.female:
                return [Gender.female]
            else:  # For non-binary/other, randomly choose one
                return random.choice([[Gender.male], [Gender.female]])

@dataclass
class UserProfile:
    user_id: str # not important for encoding
    name: str # not important for encoding
    age: int
    grade: Grade
    gender: Gender
    genders_of_interest: List[Gender]
    ethnicity: List[Ethnicity]
    faculty: Faculty
    major: List[str]
    bio: str
    preferences: str
    cluster_id: int
    fake: bool
    matches: List['UserProfile']

    def __init__(self, user_id, name, age, gender, genders_of_interest, grade, ethnicity, faculty, major, bio, preferences='',fake=True):
        # all these fields must be instantiated
        self.user_id = user_id
        self.name = name
        self.age = age
        self.gender = gender
        self.genders_of_interest = genders_of_interest
        self.grade = grade
        self.ethnicity = ethnicity
        self.faculty = faculty
        self.major = major
        self.bio = bio
        self.preferences = preferences
        self.fake = fake # put False once we onboard synthethic data
        self.matches = [] # starts empty and gets heapadded by their match_score        # except for cluster
        self.cluster_id = -1 # cluster is initialized as None before clustering

    def tostring(self):
        return (
            f"user_id: {self.user_id}, "
            f"name: {self.name}, "
            f"age: {self.age}, "
            f"grade: {self.grade.name}, "
            f"grade: {self.gender.name}, "
            f"genders of interest: {','.join(g.name for g in self.genders_of_interest)}, "
            f"ethnicity: {', '.join(e.name for e in self.ethnicity)}, "
            f"faculty: {self.faculty.name}, "
            f"major: {', '.join(self.major)}, "
            f"bio: {self.bio}, "
            f"preferences: {self.preferences},"
            f"cluster_id: {self.cluster_id} "
            f"fake: {self.fake}"
        )

# Note: major and faculty might not be necessarily related (ex.SE - CS >> Civil Eng - SE)
# might need to leverage LLM or any algorithm with Cosine Similarity
