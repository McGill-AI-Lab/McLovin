import torch
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from core.profile import UserProfile
import numpy as np
import os
from dotenv import load_dotenv
from typing import Dict, List, Tuple

load_dotenv(dotenv_path='.env')

class MatchingSystem:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
        self.index = self.pc.Index('matching-index')

    def compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two text fields using SBERT"""
        emb1 = self.model.encode(text1)
        emb2 = self.model.encode(text2)
        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    def compute_metadata_similarity(self, profile1: Dict, profile2: Dict) -> float:
        """Compute similarity score based on metadata fields"""
        score = 0.0
        weights = {
            'age': 0.1,
            'grade': 0.2,
            'faculty': 0.2,
            'major': 0.3,
            'ethnicity': 0.2
        }

        # Age similarity (inverse of difference)
        age_diff = abs(profile1['age'] - profile2['age'])
        score += weights['age'] * (1 - min(age_diff / 10, 1))  # Normalize by 10 years

        # Grade similarity
        score += weights['grade'] * (1 if profile1['grade'] == profile2['grade'] else 0.5)

        # Faculty similarity
        score += weights['grade'] * (1 if profile1['faculty'] == profile2['faculty'] else 0.5)

        # Major similarity (any overlap)
        major_overlap = len(set(profile1['major']).intersection(set(profile2['major'])))
        score += weights['major'] * (major_overlap > 0)

        # Ethnicity preference match
        ethnicity_match = any(eth in profile2['ethnicity'] for eth in profile1['ethnicity'])
        score += weights['ethnicity'] * ethnicity_match

        return score

    def match_profiles(self, user_profile: UserProfile) -> List[Tuple[str, float]]:
        """Find matches for a user considering both preferences and profile similarity"""

        # Get bio embedding for the user
        bio_embedding = self.model.encode(user_profile.bio, convert_to_numpy=True)

        # Query Pinecone for similar bios
        query_response = self.index.query(
            vector=bio_embedding[0].tolist(),
            namespace="bios-namespace",
            top_k=50,  # Get more candidates for reranking
            include_metadata=True
        )

        matches = []
        for match in query_response.matches:
            candidate_id = match.id
            candidate_metadata = match.metadata

            if candidate_id == user_profile.user_id:
                continue  # Skip self-matching

            # Compute different similarity scores
            bio_similarity = match.score  # Already computed by Pinecone

            # Compare user's preferences with candidate's attributes
            preference_similarity = self.compute_text_similarity(
                user_profile.preferences,
                f"Age {candidate_metadata['age']}, {candidate_metadata['grade']}, " +
                f"{candidate_metadata['faculty']}, {', '.join(candidate_metadata['major'])}"
            )

            # Compare metadata fields
            metadata_similarity = self.compute_metadata_similarity(
                asdict(user_profile),
                candidate_metadata
            )

            # Compute final score (weighted average)
            final_score = (
                0.4 * bio_similarity +
                0.3 * preference_similarity +
                0.3 * metadata_similarity
            )

            matches.append((candidate_id, final_score))

        # Sort by final score
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches[:10]  # Return top 10 matches
