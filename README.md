#  McLovin🎓❤️

A machine learning-powered dating application specifically designed for McGill University students. The app uses advanced matching algorithms including k-means clustering and vector embeddings to connect students based on their academic interests, faculty, and personal preferences.

## 🌟 Features

- **Smart Matching**: Uses ML algorithms to find compatible matches
- **Faculty-Aware**: Considers academic backgrounds in matching
- **Bio Analysis**: Advanced text analysis of user bios for better matching
- **Real-time Chat**: (To be implemented)
- **Profile Verification**: (To be implemented)
- **Event Matching**: (To be implemented)

## 🛠️ Tech Stack

- **Backend**: Django/Python
- **ML Framework**: PyTorch
- **Vector Database**: Weaviate
- **Primary Database**: MongoDB
- **Containerization**: Docker
- **Authentication**: (To be implemented)

## 📋 Prerequisites

- Docker & Docker Compose
- Python 3.9+
- MongoDB
- Weaviate

## 🚀 Getting Started

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/mcgill-dating-app.git
cd mcgill-dating-app
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configurations
```

3. **Build and run with Docker**
```bash
docker-compose build
docker-compose up -d
```

4. **Access the application**
- API: http://localhost:8000
- Weaviate Console: http://localhost:8080
- MongoDB: localhost:27017

## 🏗️ Project Structure

```
mcgill-dating-app/
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.ml
├── src/
│   ├── api/          # Django API
│   ├── core/         # Core functionality
│   ├── ml/           # Machine Learning
│   ├── users/        # User management
│   └── utils/        # Utilities
├── tests/
└── docker-compose.yml
```

## 💻 Development

### Setting Up Development Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests (To be implemented)
```bash
python -m pytest
```

## 🤝 Contributing (To be implemented)

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📊 ML Model Details

### Profile Matching
- Uses k-means clustering for initial grouping
- Employs vector embeddings for bio analysis
- Considers faculty and major compatibility
- Detailed algorithm documentation (To be implemented)

### Data Processing
- Bio text processing using PyTorch
- Major compatibility scoring
- Faculty weighting system
- Data pipeline documentation (To be implemented)

## 🔐 Security (To be implemented)

- Authentication system
- Data encryption
- Privacy measures
- Profile verification

## 📱 API Documentation (To be implemented)

- Authentication endpoints
- Profile management
- Matching system
- Chat system

## 🎯 Roadmap

- [ ] Implement basic matching algorithm
- [ ] Add user authentication
- [ ] Develop chat system
- [ ] Add profile verification
- [ ] Implement event matching
- [ ] Add recommendation system
- [ ] Develop mobile app

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Team

- Lead Developer: William
- Developers : Oscar, Dory, Oliver

## 📧 Contact

For any queries regarding the project, please reach out to:
- Email: william dot lafond at mail dot mcgill dot ca
- Discord: Water Fountain (or Will if you are in the discord)

## 🙏 Acknowledgments

- McGill University Faculty
- Contributors and Maintainers
- Open Source Community
