import torch
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict
import numpy as np

class BioEmbedder:
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.max_length = max_length

    def embed_bio(self, bio: str) -> np.ndarray:
        """
        Create embedding for a single bio using BERT
        """
        # Prepare the input
        inputs = self.tokenizer(
            bio,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(self.device)

        # Get BERT embedding
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use [CLS] token embedding as bio representation
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        return embedding[0]  # Return as 1D array

    def update_embeddings(self, weaviate_client, user_id: str, bio: str):
        """
        Update user's bio embedding in Weaviate
        """
        embedding = self.embed_bio(bio)

        # Store in Weaviate
        weaviate_client.store_profile_embeddings(
            user_id=user_id,
            bio_embedding=embedding.tolist()
        )

        return embedding

# Usage example:
"""
embedder = BioEmbedder()

# Process single bio
bio = "Computer Science student interested in AI and machine learning..."
embedding = embedder.embed_bio(bio)

# Store in Weaviate
embedder.update_embeddings(weaviate_client, "user123", bio)
"""

if __name__ == "__main__":
    # Process single bio
    embedder = BioEmbedder()
    bio = "Computer Science student interested in AI and machine learning..."
    embedding = embedder.embed_bio(bio)
    print(embedding)

    #embedder.update_embeddings(weaviate_client, "testuser", bio)
