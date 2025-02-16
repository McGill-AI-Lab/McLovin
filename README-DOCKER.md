# Docker Instructions

### Build the Docker image
```bash
docker build -t mclovin .
```

### Run the container
```bash
docker run -d -p 8000:8000 --name mclovin_container mclovin
```

### Access the application
Open your browser and visit: http://localhost:3000

### Access container shell (if needed)
```bash
docker exec -it mclovin_container /bin/sh
```
