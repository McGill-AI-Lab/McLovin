from django.db import models
from pymongo import MongoClient
from datetime import datetime
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager


# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
db = client["user_management"]
users_collection = db["users"]

# this app was specifically made to test the database interactions of Django with the mongodb DB

class User(models.Model):
    
    # because the default __str__ method called when you print an instance of User is not user friendly (you would get <User: User object (1)> for ex), override it to print relevant information
    def __str__(self):
        return f"musician : id = {self.id}"
    
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    birth = models.DateField(default='2000-01-01')
    email = models.EmailField(max_length=254,default='default@example.com')
    password = models.CharField(max_length=50,default="password")

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

class UserProfile(AbstractBaseUser):
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