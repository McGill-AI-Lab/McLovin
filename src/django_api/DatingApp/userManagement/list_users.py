from pymongo import MongoClient


MONGO_URI = "localhost"
client = MongoClient(MONGO_URI)
db = client["user_management"]  # Replace with your database name
users_collection = db["users"]    # Replace with your collection name

# Fetch all users
def list_users():
    users = users_collection.find()  # Retrieve all documents
    user_list = []
    for user in users:
        user_info = {
            "id": str(user["_id"]),  # Convert ObjectId to string
            "username": user.get("username", "N/A"),
            "email": user.get("email", "N/A"),  # Adjust fields as per your schema
        }
        user_list.append(user_info)
    return user_list

# Call the function and display users
all_users = list_users()
for user in all_users:
    print(user)
