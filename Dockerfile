# Use Ubuntu-based Python image
FROM python:3.11-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY src/ /app/src/
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set PYTHONPATH to include src/
ENV PYTHONPATH="${PYTHONPATH}:/app/src"

# Default command to run the application (modify as needed)
WORKDIR /app/src/django_api/DatingApp/

EXPOSE 8000

CMD ["python", "manage.py", "runserver"]
