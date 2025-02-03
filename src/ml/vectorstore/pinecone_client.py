from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Optional
import os
from dataclasses import asdict
from core.profile import UserProfile

class PineconeClient:
    self.index

    def store_profile_embeddings(self, user_id, bio, metadata):
        self.index.upsert
