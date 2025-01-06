from pymongo import MongoClient
from datetime import datetime
import bcrypt

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["user_management"]
users_collection = db["users"]

# Create a new user
def create_user(username, email, plain_password):
    # Hash the password
    hashed_password = bcrypt.hashpw(plain_password.encode('utf-8'), bcrypt.gensalt())
    
    # Construct the user document
    user = {
        "_id": username,
        "username": username,
        "email": email,
        "password": hashed_password.decode('utf-8'),
        "created_at": datetime.utcnow(),
        "roles": ["user"],
        "metadata": {
            "last_login": None,
            "profile_complete": False
        }
    }
    
    # Insert into the collection
    users_collection.insert_one(user)
    print(f"User {username} created successfully!")

# Example usage
create_user("example_user", "user@example.com", "securepassword123")
