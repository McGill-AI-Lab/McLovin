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

![kmeans_evaluation](https://github.com/user-attachments/assets/67ed247e-b774-45d3-af26-fdf9f630a9df)

Here are the results for the k-means clustering. We can see that for our initial number of 100 users, the silhouettee score peaks at 3 clusters. However, we are planning on recalibrating this with 500 fake usere, a more realistic amount of McGill students. Thus, we are projecting that silhouette score to change, and k to increase.

## Synthetic Data Generation
To validate the clustering algorithm with a larger user pool, we implemented a Bio Generator that creates realistic user profiles:

![image](https://github.com/user-attachments/assets/da889986-2de1-4d49-ac67-6de2bbf6caf5)
Here is a sample of the first rows that were generated

### Bio Generator Implementation
```
helpers/ProfileGenerator/
├── bio_generator.py        # Main generation script
├── JSON_manipulation.py    # Helper for JSON operations
└── json_prompts/          # Profile generation templates
    ├── hobbies.json       # Major-specific and general hobbies
    ├── phys_traits.json   # Physical traits by gender
    ├── pers_traits.json   # Personality traits by gender
    └── prompts.json       # Bio generation prompts
```

### Features
- **Diverse Profile Generation**: Creates varied user profiles with:
  - Major selection (11 faculty categories)
  - Hobby generation (major-specific and general)
  - Gender distribution
  - Physical and personality traits
  - AI-generated bios using Gemini

### Usage
```bash
# From project root
python helpers/ProfileGenerator/bio_generator.py
```

### Configuration
- Daily API limit: 1300 profiles
- Rate limiting: 15 profiles per minute
- Outputs saved to: `outputs/profiles.csv`
- Profile attributes:
  - gender
  - major
  - hobbies
  - physical traits
  - personality traits
  - generated bio

### Sample Output
```csv
gender,major,hobbies,Attractive phys traits,Attractive pers traits,bios
female,Engineering,rock climbing, coding,athletic build, bright smile,creative, ambitious,"Engineering student by day, rock climbing enthusiast by night..."
```

This implementation helped validate the clustering algorithm by:
1. Providing a large, diverse dataset
2. Maintaining realistic distributions of majors and interests
3. Creating natural language bios for embedding testing
4. Simulating real-world user profile variations

## *I want to try out the k-means :)*
If you’d like to run the K-Means yourself...

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


# v.02 - Image Generation
The image generation feature uses a combination of deep learning and generative AI to create natural language descriptions of user profile photos.

## Implementation Details
### 1. Face Attribute Detection
- **Model**: Custom ConvNet trained on CelebA dataset
- **Architecture**:
  - 2 Convolutional layers (64 -> 128 channels)
  - MaxPooling layers
  - 3 Fully connected layers
  - Output: 26 binary classifications
- **Attributes**: Detects 26 facial features including:
  - Gender (Male/Female)
  - Hair properties (Black, Blond, Brown, Gray, Bald)
  - Facial features (Big Nose, Pointy Nose, Big Lips)
  - Expressions (Smiling)
  - Accessories and style (Heavy_Makeup, No_Beard)

### 2. Text Generation
- **Model**: Google's Gemini 1.5 Flash
- **Implementation**:
  - Takes detected attributes as input
  - Generates 1-3 concise, natural sentences
  - Maintains consistent tone and style

## Usage
1. **Set Up Environment**
   ```bash
   # Add your API key to .env
   celeba_key=<YOUR-GEMINI-API-KEY>
   ```

2. **Train the Model** (optional, pre-trained model available)
   ```bash
   # From project root
   python src/ml/image_description/train.py
   ```

3. **Generate Descriptions**
   ```bash
   # Process single image
   python src/ml/image_description/predict.py --image_path path/to/image.jpg
   ```

## Directory Structure
```
src/ml/image_description/
├── config.py           # Configuration settings
├── models/
│   └── conv_net.py    # CNN model architecture
├── data/
│   └── celeba_dataset.py  # Dataset handling
├── train.py           # Training script
├── predict.py         # Prediction script
└── description_generator.py  # Gemini integration
```

## Model Performance
- **Attribute Detection**: ~85% accuracy on validation set
- **Processing Time**: ~0.5s per image
- **Description Generation**: ~1s per text generation

## Limitations
- Works best with front-facing portraits
- Requires clear, well-lit images
- May show bias based on CelebA dataset demographics

# Extra Stuff
## 📱 API, Security, Contributing
All to be implemented (authentication, endpoints, verification, etc.). PRs welcome!

## 🎯 Roadmap
- [x] Basic matching algorithm
- [x] User authentication
- [x] Personalized matching with CNN
- [x] Profile verification
- [ ] Chat system
- [ ] Mobile app

## 🔒 Privacy & Data Disclaimer

**Data Collection & Usage**
- **Consent:** By using McLovin, users agree to allow the application to process their personal information (e.g., profile data, images) for matchmaking purposes.
- **Image Processing:** Any face recognition or attribute detection is done in a secure environment, and images are not stored permanently without explicit consent.
- **Anonymization:** Collected data (profile preferences, embeddings, etc.) is aggregated and anonymized where possible to protect individual identities.
- **User Control:** Users can request profile deletion or opt out of image-based matching at any time.

**Security & Compliance**
- **Limited Access:** Only authorized contributors have access to sensitive data. Credentials and API keys are kept in `.env` files and are not committed to the repository.
- **No Third-Party Sharing:** McLovin does not sell or share personal data with advertisers or external parties.
- **Research & Improvements:** Data may be used internally to improve algorithms, but solely in anonymized form.

**Potential Biases**
- **Model Limitations:** Our AI models (including CNNs and clustering) may reflect biases present in training datasets (e.g., CelebA). We continuously refine our models to mitigate such biases.

> **Disclaimer:** This is a non-commercial, student-run project aimed at research and learning purposes. Use at your own discretion.


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
