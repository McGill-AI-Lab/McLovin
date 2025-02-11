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
        """Get both bio and preference embeddings for a user using fetch"""
        try:
            logging.info(f"Fetching embeddings for user {user_id}")

            # Fetch bio embedding
            bio_result = self.index.fetch(
                ids=[user_id],
                namespace="bio-embeddings"
            )

            # Fetch preference embedding
            pref_result = self.index.fetch(
                ids=[user_id],
                namespace="preferences-embeddings"
            )

            # Check if vectors exist in both namespaces
            if user_id not in bio_result['vectors']:
                logging.error(f"No bio embedding found for user {user_id}")
                return np.zeros(384), np.zeros(384)

            if user_id not in pref_result['vectors']:
                logging.error(f"No preference embedding found for user {user_id}")
                return np.zeros(384), np.zeros(384)

            # Extract the embeddings
            bio_embedding = np.array(bio_result['vectors'][user_id]['values'])
            pref_embedding = np.array(pref_result['vectors'][user_id]['values'])

            logging.info(f"Successfully retrieved embeddings for user {user_id}")
            return bio_embedding, pref_embedding

        except Exception as e:
            logging.error(f"Error getting embeddings for user {user_id}: {str(e)}")
            return np.zeros(384), np.zeros(384)

    def calculate_match_score_cached(self, user1_id: str, user2_id: str,
                                   user1_data: Dict, user2_data: Dict) -> float:
        """Calculate match score using cached embeddings"""
        try:
            # Check compatibility using metadata
            if not self.is_compatible(
                {'id': user1_id, 'metadata': user1_data['metadata']},
                {'id': user2_id, 'metadata': user2_data['metadata']}
            ):
                return 0.0

            # Calculate bidirectional scores using cached embeddings
            score1 = cosine_similarity(
                user1_data['pref'].reshape(1, -1),
                user2_data['bio'].reshape(1, -1)
            )[0][0]

            score2 = cosine_similarity(
                user2_data['pref'].reshape(1, -1),
                user1_data['bio'].reshape(1, -1)
            )[0][0]

            return (score1 + score2) / 2

        except Exception as e:
            logging.error(f"Error calculating match score: {str(e)}")
            return 0.0

    def find_matches_in_cluster(self, cluster_id: float) -> List[Tuple]:
        """Find optimal matches within a single cluster"""
        try:
            # Get only users from this specific cluster
            cluster_users = self.index.query(
                vector=[0.0] * 384,
                filter={"cluster_id": cluster_id},
                namespace="bio-embeddings",
                include_metadata=True,
                top_k=1000
            )

            users = getattr(cluster_users, 'matches', [])
            num_users = len(users)

            logging.info(f"\nProcessing Cluster {cluster_id}:")
            logging.info(f"Number of users in cluster: {num_users}")

            if num_users < 2:
                logging.info(f"Skipping cluster {cluster_id} - needs at least 2 users")
                return []

            # Pre-fetch all embeddings for the cluster
            user_embeddings = {}  # Cache for embeddings
            for user in users:
                user_id = user['id']
                bio_embedding, pref_embedding = self.get_user_embeddings(user_id)
                user_embeddings[user_id] = {
                    'bio': bio_embedding,
                    'pref': pref_embedding,
                    'metadata': user['metadata']
                }

            # Create graph
            G = nx.Graph()

            # Add nodes
            for user in users:
                G.add_node(user['id'], **user['metadata'])

            # Calculate possible combinations
            possible_pairs = (num_users * (num_users - 1)) // 2
            logging.info(f"Processing {num_users} users ({possible_pairs} possible pairs)")

            # Add edges with weights (matching scores)
            edge_count = 0
            processed_pairs = 0

            # Process each possible pair exactly once
            for i, user1 in enumerate(users):
                user1_id = user1['id']
                for j in range(i + 1, len(users)):
                    user2 = users[j]
                    user2_id = user2['id']
                    processed_pairs += 1

                    # Calculate matching score using cached embeddings
                    score = self.calculate_match_score_cached(
                        user1_id, user2_id,
                        user_embeddings[user1_id],
                        user_embeddings[user2_id]
                    )

                    if score > 0:
                        G.add_edge(user1_id, user2_id, weight=score)
                        edge_count += 1
                        logging.debug(f"Match found: {user1_id} - {user2_id} (score: {score:.3f})")

            logging.info(f"Processed {processed_pairs} pairs, found {edge_count} compatible matches")

            if edge_count == 0:
                logging.info(f"No compatible matches in cluster {cluster_id}")
                return []

            # Find optimal matching
            optimal_matches = nx.max_weight_matching(G, maxcardinality=True)

            # Convert to list of tuples with scores
            match_list = []
            for user1_id, user2_id in optimal_matches:
                score = G[user1_id][user2_id]['weight']
                match_list.append((user1_id, user2_id, score))
                logging.info(f"Optimal match: {user1_id} - {user2_id} (score: {score:.3f})")

            logging.info(f"Created {len(match_list)} final matches in cluster {cluster_id}")
            return match_list

        except Exception as e:
            logging.error(f"Error processing cluster {cluster_id}: {str(e)}")
            return []

    def run_matching(self):
        """Run matching algorithm cluster by cluster"""
        logging.info("Starting matching process...")

        # Get all users
        all_users = self.index.query(
            vector=[0.0] * 384,
            filter={},
            namespace="bio-embeddings",
            include_metadata=True,
            top_k=10000
        )

        # Get unique cluster IDs, handling potential missing values
        cluster_ids = set()
        for user in getattr(all_users, 'matches'):
            try:
                cluster_id = float(user['metadata'].get('cluster_id', -1))
                if cluster_id >= 0:  # Only add valid clusters
                    cluster_ids.add(cluster_id)
            except (KeyError, ValueError) as e:
                logging.warning(f"Could not get cluster_id for user: {user.get('id', 'unknown')}")
                continue

        logging.info(f"Found {len(cluster_ids)} valid clusters: {sorted(cluster_ids)}")

        # Process each cluster separately
        all_matches = []
        for cluster_id in sorted(cluster_ids):
            logging.info(f"\nProcessing cluster {cluster_id}")
            cluster_matches = self.find_matches_in_cluster(cluster_id)
            all_matches.extend(cluster_matches)

            # Update matches in database
            if cluster_matches:
                self.update_matches(cluster_matches)

        # Summary of all matches
        logging.info("\nMatching Summary:")
        logging.info(f"Total clusters processed: {len(cluster_ids)}")
        logging.info(f"Total matches generated: {len(all_matches)}")

        return all_matches


    def update_matches(self, matches: List[Tuple]):
        """Update user records with their matches"""
        for user1_id, user2_id, score in matches:
            # Store match data (convert score to string for Pinecone metadata)
            match_data = {
                "matched_users": [user2_id],
                "match_scores": [f"{score:.4f}"]
            }

            # Update both namespaces
            for namespace in ["bio-embeddings", "preferences-embeddings"]:
                # Update user1's matches
                self.index.update(
                    id=user1_id,
                    namespace=namespace,
                    set_metadata=match_data
                )

                # Update user2's matches
                self.index.update(
                    id=user2_id,
                    namespace=namespace,
                    set_metadata={
                        "matched_users": [user1_id],
                        "match_scores": [f"{score:.4f}"]
                    }
                )

    def check_index_content(self):
        """Check the content of the Pinecone index"""
        try:
            # Get stats for both namespaces
            bio_stats = self.index.describe_index_stats(namespace="bio-embeddings")
            pref_stats = self.index.describe_index_stats(namespace="preferences-embeddings")

            logging.info(f"Bio embeddings namespace stats: {bio_stats}")
            logging.info(f"Preference embeddings namespace stats: {pref_stats}")

            # Try to fetch a sample from each namespace by querying first
            sample_query = self.index.query(
                vector=[0.0] * 384,
                namespace="bio-embeddings",
                include_metadata=True,
                top_k=1
            )

            if getattr(sample_query, 'matches'):
                sample_id = sample_query.matches[0]['id']

                # Now fetch the complete records for this ID
                bio_sample = self.index.fetch(
                    ids=[sample_id],
                    namespace="bio-embeddings"
                )

                pref_sample = self.index.fetch(
                    ids=[sample_id],
                    namespace="preferences-embeddings"
                )

                logging.info(f"Sample bio record: {bio_sample}")
                logging.info(f"Sample preference record: {pref_sample}")
            else:
                logging.error("No samples found in index")

        except Exception as e:
            logging.error(f"Error checking index content: {str(e)}")
