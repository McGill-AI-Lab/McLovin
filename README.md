# McLovin🎓❤️

A ML-powered dating application for McGill students. It uses advanced algorithms (k-means clustering, vector embeddings) to connect students based on shared academic interests, faculty, and personal preferences.

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

## 🚀 Getting Started
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

## 🏗️ Project Structure
```
mcgill-dating-app/
├── docker/
│   └── (Dockerfiles and scripts for containerization)
├── src/
│   ├── frontend/  # TBD later
│   ├── api/       # Django API (views, URLs, serializers, REST endpoints)
│   ├── core/      # Core functionality (matching logic, key app classes, services)
│   ├── ml/        # Machine Learning code (model definitions, embeddings, etc.)
│   ├── users/     # User management (auth, registration, user model, etc.)
│   └── utils/     # General-purpose utilities (helpers, validators, etc.)
├── tests/
│   └── (Test files for unit, integration, etc.)
└── docker-compose.yml
```

## 💻 Development
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```
**Tests** (to be implemented):
```bash
python -m pytest
```

## 📊 ML Model Details
- **k-means clustering** for initial grouping
- **Vector embeddings** for bio analysis
- **Faculty / Major weighting**

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
