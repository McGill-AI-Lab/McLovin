from pymongo import MongoClient
from functools import wraps
from django.http import JsonResponse
from django.shortcuts import redirect
from dataclasses import dataclass, fields
import sys
from pymongo import MongoClient
from bson.objectid import ObjectId
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.append('../../') # assuming working dir is DatingApp (with manage.py), this adds the core directory to PATH

from core.profile import UserProfile,Grade, Faculty,Ethnicity
from ml.clustering.cluster_users import cluster_users
from core.embedder import Embedder

from dataclasses import dataclass, asdict

# defining a decorator to check if an user is authenticated
from functools import wraps
from urllib.parse import urlencode
from django.shortcuts import redirect
from dotenv import find_dotenv,dotenv_values
from bson import ObjectId

config_path = find_dotenv("config.env")
config = dotenv_values(dotenv_path=config_path) # returns a dictionnary of dotenv values

from bson import ObjectId

enum_fields = {
    "grade": Grade,
    "ethnicity": Ethnicity,
    "faculty": Faculty
}

def serialize_enums(data):
    """Recursively convert enums inside lists/dicts to their string names."""
    if isinstance(data, list):  # If data is a list, serialize each item
        return [serialize_enums(item) for item in data]
    elif isinstance(data, dict):  # If data is a dictionary, serialize each value
        return {key: serialize_enums(value) for key, value in data.items()}
    elif isinstance(data, Grade) or isinstance(data, Faculty) or isinstance(data, Ethnicity):
        return data.name
    return data  # Return as is if not an enum

def deserialize_enums(data, enum_mapping=enum_fields):
    """
    Recursively convert stored enum strings (including lists) back to their respective Enums.

    :param data: The data retrieved from MongoDB.
    :param enum_mapping: A dictionary mapping field names to Enum classes.
    :return: Data with Enums restored.
    """
    if isinstance(data, list):  # If data is a list, deserialize each item
        return [deserialize_enums(item, enum_mapping) for item in data]
    elif isinstance(data, dict):  # If data is a dictionary, deserialize each value
        return {key: deserialize_enums(value, enum_mapping) for key, value in data.items()}
    elif isinstance(data, str):  # If it's a string, check if it belongs to any Enum
        for enum_cls in enum_mapping.values():
            try:
                return enum_cls[data]  # Convert string to Enum
            except KeyError:
                continue
    return data  # Return as is if not an enum




def login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Check if the user is authenticated
        if not request.session.get("user_id"):
            # Capture the origin URL (current path and query string)
            origin = request.get_full_path()

            # Add the `next` parameter with the origin URL
            query_params = urlencode({"next": origin})
            login_url = f"/login/?{query_params}"

            # Redirect to the login URL with the `next` parameter
            return redirect(login_url)

        return view_func(request, *args, **kwargs)
    return wrapper


# Connect to MongoDB

client = MongoClient(config["MONGO_URI"])
db = client[config["MONGO_DB_NAME"]]
users_collection = db["users"]

# this app was specifically made to test the database interactions of Django with the mongodb DB
class User(UserProfile):
    """A custom User class for MongoDB interaction using pymongo."""

    def __init__(self, db_uri=config["MONGO_URI"], db_name=config["MONGO_DB_NAME"], collection_name="users"):
        self.client = MongoClient(db_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.to_dict_with_defaults()

    def create_user(self,user_data:dict):
        """
        Input : a dict of user data (json like)
        Output: the user mongodb ID
        Creates a new user in the database."""
        if not user_data["email"] or not user_data["password"]:
            raise ValueError("Email and password are required")

        user_data["password"] = generate_password_hash(user_data["password"]) # hash the password

        try:
            # Insert the user data into the collection
            result = self.collection.insert_one(user_data)
            return result.inserted_id  # Return the ID of the inserted document
        except Exception as e:
            print(f"Error inserting user: {e}")
            return None

    def get_user_by_id(self, user_id):
        """Fetch a user by their ID and converts all enum strings inside lists back to Enums."""
        user_data = self.collection.find_one({"_id": ObjectId(user_id)})

        if not user_data:
            return None

        enum_fields = {
            "grade": Grade,
            "ethnicity": Ethnicity,
            "faculty": Faculty
        }

        user_data = deserialize_enums(user_data, enum_fields)

        print("RAW USER DATA RETRIEVED:", user_data)
        return user_data

    def authenticate_user(self, email, password):
        """Authenticates a user with email and password."""
        user = self.collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            return user # collection with all attributes etc
        return None

    def reset_password(self, email):
        # no implementation yet : add email token reset ? Or security question ?
        pass

    def update_user(self, user_id, updates):
        """Updates user details.
        Input : a user_id (unique to every user) and a dictionary of updates (python-like)
        Output : result of the update
        """
        return self.collection.update_one({"_id": ObjectId(user_id)}, {"$set": updates})

    def delete_user(self, user_id):
        """Deletes a user from the database."""
        return self.collection.delete_one({"_id": ObjectId(user_id)})

    def to_dict_with_defaults(self):
        type_defaults = {str: "", int: 0, float: 0.0, bool: False,list:[]}
        return {field.name: type_defaults.get(field.type, []) for field in fields(self)}
    # initialize the dictionnary from the profile dataclass

    def get_enum_options(self):
        dico = {}
        for enum in Grade,Faculty,Ethnicity:
            dico[enum.__name__] = [e.name for e in enum]
        return dico

    def perform_matchmaking(self,user_id):
        """
        # assigns cluster, gets a match
        """

    def __str__(self):
        return f"MongoDB User Collection: {self.collection.name}"

    def embed(self,user_collection):
        """
        Embeds a user's profile with the embeddings model
        :param user_id:
        :return:
        """
        # use _to_dict_with_defaults first to send to the frontend
        embedder = Embedder()
        # get the user's dictionnary of data from the MongoDB database
        # create a UserProfile object from the dictionary
        print("user collection being sent to Embedder",user_collection)

        try :
            embedding = embedder.process_profile(user_collection)

            print("result from embedder", embedding)

        except Exception as e:
            print("An error occured during the process of user embedding : ",e)

        print("Profile stored in Pinecone")

        try:
            print(cluster_users(10, 101))
        
        except Exception as e:
            print("An error occured during the process of clustering : ",e)

        print("users successfully clustered !")

class Matching(User):

    def assign(self,user_id):
        """
        Assigns a user to a cluster

        :param user_id:
        :return:
        """
        #cluster_users()
        pass


#TODO Implement the UserManager for login and signup views
