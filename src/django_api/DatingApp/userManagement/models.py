from djongo import models
from pymongo import MongoClient
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from functools import wraps
from django.http import JsonResponse
from django.shortcuts import redirect
import sys
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime
from django.utils.crypto import get_random_string
from django.core.mail import send_mail
from werkzeug.security import generate_password_hash, check_password_hash

sys.path.append('../../') # assuming working dir is DatingApp (with manage.py), this adds the core directory to PATH

#from core.profile import UserProfile

# defining a decorator to check if an user is authenticated
from functools import wraps
from urllib.parse import urlencode
from django.shortcuts import redirect

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
client = MongoClient("mongodb://localhost:27017")
db = client["user_management"]
users_collection = db["users"]
#"mongodb://localhost:27017/"

# this app was specifically made to test the database interactions of Django with the mongodb DB
class User:
    """A custom User class for MongoDB interaction using pymongo."""

    def __init__(self, db_uri="your_db_uri", db_name="your_db_name", collection_name="users"):
        self.client = MongoClient(db_uri)  # Replace with your MongoDB URI
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def create_user(self, email, password, first_name="", last_name="", birth_date=None):
        """Creates a new user in the database."""
        if not email or not password:
            raise ValueError("Email and password are required")

        hashed_password = generate_password_hash(password)
        user_data = {
            "email": email,
            "password": hashed_password,
            "first_name": first_name,
            "last_name": last_name,
            "birth_date": birth_date or datetime(2000, 1, 1).isoformat(),
            "is_active": True,
            "created_at": datetime.utcnow(),
            "cluster_id": -1,
        }

        result = self.collection.insert_one(user_data)
        return str(result.inserted_id)

    def get_user_by_id(self, user_id):
        """Fetches a user by their ID."""
        return self.collection.find_one({"_id": ObjectId(user_id)})

    def authenticate_user(self, email, password):
        """Authenticates a user with email and password."""
        user = self.collection.find_one({"email": email})
        if user and check_password_hash(user["password"], password):
            return user
        return None

    def reset_password(self, email):
        """Generates a password reset token and sends a reset email."""
        user = self.collection.find_one({"email": email})
        if not user:
            raise ValueError("User not found")

        token = get_random_string(length=32)
        reset_link = f"http://yourdomain.com/reset-password/{token}/"
        send_mail(
            "Password Reset Request",
            f"Click the link to reset your password: {reset_link}",
            "no-reply@yourdomain.com",
            [email],
        )

        # Store the token with an expiration time in the database
        self.collection.update_one(
            {"_id": user["_id"]},
            {"$set": {"reset_token": token, "reset_token_expires": datetime.utcnow()}}
        )
        return token

    def update_user(self, user_id, updates):
        """Updates user details.
        Input : a user_id (unique to every user) and a dictionary of updates (python-like)
        Output : result of the update
        """
        return self.collection.update_one({"_id": ObjectId(user_id)}, {"$set": updates})

    def delete_user(self, user_id):
        """Deletes a user from the database."""
        return self.collection.delete_one({"_id": ObjectId(user_id)})

    def __str__(self):
        return f"MongoDB User Collection: {self.collection.name}"
#TODO Implement the UserManager for login and signup views
"""
class UserManager(BaseUserManager):
    def create_user(self, email, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        return self.create_user(email, password, **extra_fields)

class UserProfileDD(AbstractBaseUser):
    email = models.EmailField(unique=True)
    first_name = models.CharField(max_length=30)
    last_name = models.CharField(max_length=30)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['first_name', 'last_name']

    objects = UserManager()

    def __str__(self):
        return self.email
"""
# models.py
from django.db import models

class Grade(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


class Ethnicity(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


class Faculty(models.Model):
    name = models.CharField(max_length=255)

    def __str__(self):
        return self.name


class UserProfile(models.Model):
    user_id = models.CharField(max_length=255)
    name = models.CharField(max_length=255)
    age = models.IntegerField()
    grade = models.ForeignKey(Grade, on_delete=models.CASCADE)
    ethnicity = models.ManyToManyField(Ethnicity)
    faculty = models.ForeignKey(Faculty, on_delete=models.CASCADE)
    major = models.JSONField()  # List of strings
    bio = models.TextField()
    preferences = models.TextField()
