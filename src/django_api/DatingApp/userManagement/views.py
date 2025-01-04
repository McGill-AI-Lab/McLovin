from django.shortcuts import render
from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import UserSerializer
from .models import login_required
from pymongo import MongoClient
import bcrypt
from django.shortcuts import redirect
import json

from dotenv import dotenv_values, find_dotenv,load_dotenv


config_path = find_dotenv("config.env")
config = dotenv_values(dotenv_path=config_path) # returns a dictionnary of dotenv values

mongo_client = MongoClient(config["MONGO_URI"])
mongo_db = mongo_client[config["MONGO_DB_NAME"]]

@api_view(['GET', 'POST'])
@csrf_exempt
def login_user(request):
    if request.method == "POST":
        data  = json.loads(request.body)
        email = data.get("email")
        password = data.get("password")

        user = mongo_db.users.find_one({"email": email})
        if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return JsonResponse({"error": "Invalid email or password"}, status=401)

        # Store user_id in session to persist authentication
        request.session["user_id"] = str(user["_id"])  # Use str to serialize MongoDB ObjectId
        return JsonResponse({"message": "Login successful"})
    elif request.method == "GET":
        return render(request,"login.html")

@api_view(['GET', 'POST'])
@login_required
@csrf_exempt
def logout_user(request):
    if request.method == "GET":
        request.session.flush()  # Clear session data
        #return JsonResponse({"message": "Logged out successfully"})
        return redirect("login")

@api_view(['GET', 'POST'])
@csrf_exempt
def signup_user(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse the JSON body
            firstName = data.get("firstName")
            lastName = data.get("lastName")
            email = data.get("email")
            password = data.get("password")

            # Basic validation
            if not firstName or not password or not lastName or not email:
                return JsonResponse({"error": "User name and password are required"}, status=400)

            # Check if the username already exists
            existing_user = mongo_db.users.find_one({"username": firstName+" "+lastName})
            existing_email = mongo_db.users.find_one({"email": email})
            if existing_user or existing_email:
                return JsonResponse({"error": "User Name or email is already taken"}, status=400)

            # Hash the password
            hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt())

            # Save the user in the database
            user = {
                "username": firstName+" "+lastName,
                "email": email,
                "password": hashed_password.decode(),  # Store the hashed password as a string
            }
            result = mongo_db.users.insert_one(user)

            return JsonResponse({
                "message": "User registered successfully",
                "user_id": str(result.inserted_id)
            })
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid JSON payload"}, status=400)
    else:
        return render(request,"signup.html")

@login_required
@csrf_exempt
def restricted_view(request):
    return render(request,"restricted.html")
    