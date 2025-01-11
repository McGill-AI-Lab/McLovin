# McLovin🎓❤️

A ML-powered dating application for McGill students. It uses advanced algorithms (k-means clustering, vector embeddings) to connect students based on shared academic interests, faculty, and personal preferences.

# Project Overview
## 🌟 Features
- **Smart Matching** via ML algorithms
- **Faculty-Aware** pairings
- **Bio Analysis** with text embeddings
- **To be added**: Chat, Profile Verification, Event Matching

## 🛠️ Tech Stack
- **Backend**: Django/Python
- **ML Framework**: PyTorch
- **Vector Database**: Pinecone
- **Primary Database**: MongoDB
- **Containerization**: Docker

## 📋 Prerequisites
- Docker & Docker Compose
- Python 3.9+
- MongoDB
- Pinecone

## 🏗️ Project Structure
```
mcgill-dating-app/
├── docker/
│   └── (Dockerfiles and scripts for containerization)
├── helpers/
│   └── (helper scripts, or any static file we use for dev/testing)
├── outputs/
│   └── (plots, images, csv, data, etc. from src/ml)
├── src/
│   ├── frontend/  # TBD later
│   ├── api/       # Django API (views, URLs, serializers, REST endpoints)
│   ├── core/      # Core functionality (matching logic, key app classes, services)
│   ├── ml/        # Machine Learning code (model definitions, embeddings, etc.)
│   │   ├── clustering/        # K-means, cluster-based logic
│   │   ├── matching/          # Matching logic / embeddings
│   │   └── image_description/ # CNN and generative text
│   ├── users/     # User management (auth, registration, user model, etc.)
│   └── helpers/     # General-purpose utilities (helpers, validators, etc.)
├── tests/
│   └── (unit and integration testing)
├── .env            # API keys are listed here in txt
└── docker-compose.yml # build tool for docker/
```

## .env should look like this...
```
PINECONE_KEY=<YOUR-KEY>
GEMINI_KEY=<YOUR-KEY>
GOOGLE_API_KEY=<YOUR-KEY>
celeba_key=<YOUR-KEY>
```

# 🚀 Getting Started
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/mcgill-dating-app.git
   cd mcgill-dating-app
   ```
2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env
   ```
3. **Build and run with Docker**
   ```bash
   docker-compose build
   docker-compose up -d
   ```
4. **Access the application**
   - API: http://localhost:8000
   - MongoDB: localhost:27017


## 💻 Development
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Python Integration Tests**:
```bash
python pytest tests
```

# 📊 v.01 - Clustering and Matching
- **Embed** user profile data via **SBERT** (PyTorch).
- **Store** embeddings in **Pinecone** (+ optional sentiment score).
- **Assign** each user to a cluster via **K-Means** (Pytorch).
- **Rank** matches within that cluster using **cosine similarity**.
- **Refine** final scores with metadata weighting

*in the end, this creates a sort of **Matching Elo** for the users per cluster*

## *I want to try out the k-means :)*
If you’d like to experiment with K-Means clustering on user bio embeddings:

### 1. Configure Pinecone and .env
Make sure your ```.env``` includes your Pinecone key:

```bash
Copy code
PINECONE_KEY=your_pinecone_api_key
and any other environment variables required by the scripts.
```

### 2. Run the Clustering Script
Inside the ```src/ml/clustering/``` folder, there’s a script called ```cluster_users.py``` which performs a simple K-Means clustering on up to 1,000 user embeddings from Pinecone. For instance:

```bash
Copy code
python src/ml/clustering/cluster_users.py
```
This will:

1. Load embeddings from your Pinecone index (named ```bio-embeddings``` by default).
2. Standardize the embeddings.
3. Perform K-Means clustering (by default, 10 clusters, seed = 101).
4. Return assignments and centroids.

You can modify the number of clusters or random state by editing ```cluster_users.py``` or adjusting the function call.

### 3. Evaluate Different k Values

The ```evaluation.py``` script helps you **find the optimal k (num of clusters)**:

```bash
Copy code
python src/ml/clustering/evaluation.py
```

This script will:

1. Query Pinecone for embeddings (like cluster_users.py).
2. Split them into train/validation sets.
3. Train multiple K-Means models (with k from 2 to 10, by default).
4. Calculate evaluation metrics (Inertia, Silhouette, Calinski-Harabasz).
5. Generate an output plot ```kmeans_evaluation.png``` in an ```outputs/``` folder.

Finally, it will pick an “optimal k” based on **silhouette score**, cluster everything, and print some basic cluster stats (size, density, etc.).

# v.02 - Image Generation (Incomplete)
- **CelebA-based Convolutional Network** classifies up to 26 facial attributes (Male, Bald, Smiling, etc.).
- **Generative AI** (using Google's GEMINI) converts these detected attributes into concise descriptive text.

*this feature gives additional precision to the bio_description inputted by user*

# Extra Stuff
## 📱 API, Security, Contributing
All to be implemented (authentication, endpoints, verification, etc.). PRs welcome!

## 🎯 Roadmap
- [ ] Basic matching algorithm
- [ ] User authentication
- [ ] Chat system
- [ ] Profile verification
- [ ] Event matching
- [ ] Recommendation system
- [ ] Mobile app

## 📄 License
MIT License - see [LICENSE](LICENSE) for details.

## 👥 Team
- **Lead Developer**: William
- **Developers**: Oscar, Dory, Oliver

## 📧 Contact
- **Email**: william dot lafond at mail dot mcgill dot ca
- **Discord**: “Water Fountain” (or “Will”)

## 🙏 Acknowledgments
- McGill University
- Open Source Community
