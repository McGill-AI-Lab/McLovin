import networkx as nx
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple
from src.core.profile import UserProfile, Gender
from pinecone import Pinecone
import os
from dotenv import load_dotenv
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

class MatchMaker:
    def __init__(self):
        load_dotenv(dotenv_path='.env')
        self.pc = Pinecone(api_key=os.getenv("PINECONE_KEY"))
        self.index = self.pc.Index("matching-index")
        self.setup_logging()

    def setup_logging(self):
        output_dir = 'outputs'
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = os.path.join(output_dir, f'matching_{timestamp}.log')

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename),
                logging.StreamHandler()
            ]
        )

    def is_compatible(self, user1: Dict, user2: Dict) -> bool:
        """Check if two users are compatible based on gender preferences"""
        try:
            user1_metadata = user1['metadata']
            user2_metadata = user2['metadata']

            # If genders_of_interest is not present, temporarily assume compatibility
            if 'genders_of_interest' not in user1_metadata or 'genders_of_interest' not in user2_metadata:
                return True

            user1_gender = Gender[user1_metadata['gender']]
            user2_gender = Gender[user2_metadata['gender']]

            user1_gois = [Gender[goi] for goi in user1_metadata['genders_of_interest']]
            user2_gois = [Gender[goi] for goi in user2_metadata['genders_of_interest']]

            if user1_gender not in user2_gois or user2_gender not in user1_gois:
                return False
            return True

        except (KeyError, ValueError) as e:
            logging.warning(f"Compatibility check failed: {str(e)}. Assuming compatible.")
            return True  # Temporary fallback

    def get_user_embeddings(self, user_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get both bio and preference embeddings for a user"""
        try:
            # Get bio embedding
            bio_result = self.index.query(
                vector=[0.0] * 384,
                filter={"user_id": user_id},
                namespace="bio-embeddings",
                include_values=True,
                top_k=1
            )

            # Get preference embedding
            pref_result = self.index.query(
                vector=[0.0] * 384,
                filter={"user_id": user_id},
                namespace="preferences-embeddings",
                include_values=True,
                top_k=1
            )

            # Add error checking
            if not getattr(bio_result, 'matches') or not getattr(pref_result, 'matches'):
                logging.error(f"No embeddings found for user {user_id}")
                return np.zeros(384), np.zeros(384)  # Return zero vectors as fallback

            bio_embedding = np.array(getattr(bio_result, 'matches')[0]['values'])
            pref_embedding = np.array(getattr(pref_result, 'matches')[0]['values'])

        except Exception as e:
            logging.error(f"Error getting embeddings for user {user_id}: {str(e)}")
            return np.zeros(384), np.zeros(384)  # Return zero vectors as fallback

        return bio_embedding, pref_embedding

    def calculate_match_score(self, user1: Dict, user2: Dict) -> float:
        """Calculate mutual matching score between two users"""
        if not self.is_compatible(user1, user2):
            return 0.0

        # Get embeddings for both users
        user1_bio, user1_pref = self.get_user_embeddings(user1['id'])
        user2_bio, user2_pref = self.get_user_embeddings(user2['id'])

        # Calculate bidirectional scores
        # How well user1's preferences match user2's bio
        score1 = cosine_similarity(
            user1_pref.reshape(1, -1),
            user2_bio.reshape(1, -1)
        )[0][0]

        # How well user2's preferences match user1's bio
        score2 = cosine_similarity(
            user2_pref.reshape(1, -1),
            user1_bio.reshape(1, -1)
        )[0][0]

        # Return average of both scores (total score)
        total_score = (score1 + score2) / 2
        return total_score

    def find_matches_in_cluster(self, cluster_id: int) -> List[Tuple]:
        """Find optimal matches within a cluster using maximum weight matching"""

        # Get all users in the cluster from bio namespace
        cluster_users = self.index.query(
            vector=[0.0] * 384,
            filter={"cluster_id": cluster_id},
            namespace="bio-embeddings",
            include_values=True,
            include_metadata=True,
            top_k=1000
        )

        if len(getattr(cluster_users, 'matches')) < 2:
            logging.info(f"Cluster {cluster_id} has less than 2 users. Skipping.")
            return []

        # Create graph for maximum weight matching
        G = nx.Graph()

        # Add all users as nodes
        for user in getattr(cluster_users,'matches'):
            G.add_node(user['id'], **user['metadata'])

        # Add edges with weights (matching scores)
        for i, user1 in enumerate(getattr(cluster_users,'matches')):
            for user2 in getattr(cluster_users,'matches')[i+1:]:
                score = self.calculate_match_score(user1, user2)
                if score > 0:  # Only add edge if users are compatible
                    G.add_edge(user1['id'], user2['id'], weight=score)

        # Find maximum weight matching
        matches = nx.max_weight_matching(G, maxcardinality=True)

        # Convert matches to list of tuples with scores
        match_list = []
        for user1_id, user2_id in matches:
            score = G[user1_id][user2_id]['weight']
            match_list.append((user1_id, user2_id, score))

        return match_list

    def run_matching(self):
        """Run matching algorithm for all clusters"""
        logging.info("Starting matching process...")

        # Get all unique cluster IDs from bio namespace
        all_users = self.index.query(
            vector=[0.0] * 384,
            filter={},
            namespace="bio-embeddings",
            include_metadata=True,
            top_k=10000
        )

        print("Number of matches:", len(getattr(all_users, 'matches')))
        cluster_ids = set()
        for user in getattr(all_users,'matches'):
            print("User ID:", user['id'])
            print("User metadata:", user['metadata'])
            print("Cluster:", user['metadata'].get('cluster_id', 'No cluster found'))
            if 'cluster_id' in user['metadata']:
                cluster_ids.add(user['metadata']['cluster_id'])


        all_matches = []
        for cluster_id in cluster_ids:
            logging.info(f"Processing cluster {cluster_id}")
            cluster_matches = self.find_matches_in_cluster(cluster_id)
            all_matches.extend(cluster_matches)

            # Update matches in database for both namespaces
            self.update_matches(cluster_matches)

        logging.info(f"Matching complete. Generated {len(all_matches)} matches.")
        return all_matches

    def update_matches(self, matches: List[Tuple]):
        """Update user records with their matches in both namespaces"""
        for user1_id, user2_id, score in matches:
            # Update metadata for both namespaces
            for namespace in ["bio-embeddings", "preferences-embeddings"]:
                # Update for user1
                self.index.update(
                    id=user1_id,
                    namespace=namespace,
                    set_metadata={"matched_users": [user2_id], "match_scores": [float(score)]}
                )

                # Update for user2
                self.index.update(
                    id=user2_id,
                    namespace=namespace,
                    set_metadata={"matched_users": [user1_id], "match_scores": [float(score)]}
                )
