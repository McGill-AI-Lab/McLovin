
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from .user import login_required
from pymongo import MongoClient
from django.http import JsonResponse
from django.shortcuts import render, redirect
import json
import bcrypt

from dotenv import dotenv_values, find_dotenv,load_dotenv

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from bson import ObjectId

config_path = find_dotenv("config.env")
config = dotenv_values(dotenv_path=config_path) # returns a dictionnary of dotenv values

mongo_client = MongoClient(config["MONGO_URI"])
mongo_db = mongo_client[config["MONGO_DB_NAME"]]
users_collection = mongo_db["users"]

@api_view(['GET', 'POST'])
@csrf_exempt
def login_user(request):
    if request.method == "POST":
        data = json.loads(request.body)
        email = data.get("email")
        password = data.get("password")

        # Fetch user from MongoDB
        user = mongo_db.users.find_one({"email": email})
        if not user or not bcrypt.checkpw(password.encode(), user["password"].encode()):
            return JsonResponse({"error": "Invalid email or password"}, status=401)

        # Store user_id in session to persist authentication
        request.session["user_id"] = str(user["_id"])  # Use str to serialize MongoDB ObjectId

        # Retrieve the `next` parameter to determine where to redirect after login
        next_url = request.GET.get("next", "%2Fhome%2F")  # Default to home if `next` is not provided

        # Redirect to the specified URL or return a success response
        return JsonResponse({"message": "Login successful", "redirect": next_url})

    elif request.method == "GET":
        # Pass the `next` parameter to the template for inclusion in the form
        next_url = request.GET.get("next", "/")
        return render(request, "login.html", {"next": next_url})


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
                'age': 0,
                'grade': 0,
                'ethnicity': "Dunno",
                'faculty': "Dunno",
                'major': "Hmm",
                'bio': "This is my bio !",
                'preferences': "None",
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

@login_required
def user_dashboard(request, user_id):
    # Retrieve user profile
    #user_profile = get_object_or_404(UserProfile, user_id=user_id)
    user_profile = users_collection.find_one({"_id": ObjectId(user_id)})

    if not user_profile:
        messages.error(request, 'User profile not found.')
        return redirect('home')

    if request.method == 'POST':
        if 'edit' in request.POST:
            # Update the fields provided by the POST request
            updated_data = {
                'username': request.POST.get('name', user_profile.get('name', '')),
                'age': int(request.POST.get('age', user_profile.get('age', 0))),
                'grade': request.POST.get('grade', user_profile.get('grade', '')),
                'ethnicity': request.POST.getlist('ethnicity') or user_profile.get('ethnicity', []),
                'faculty': request.POST.get('faculty', user_profile.get('faculty', '')),
                'major': request.POST.getlist('major') or user_profile.get('major', []),
                'bio': request.POST.get('bio', user_profile.get('bio', '')),
                'preferences': request.POST.get('preferences', user_profile.get('preferences', '')),
            }

            # Update the user profile in MongoDB
            mongo_db.users.update_one({'_id': ObjectId(user_id)}, {'$set': updated_data})
            messages.success(request, 'Profile updated successfully!')
            return redirect('user_dashboard', user_id=user_id)

        elif 'delete' in request.POST:
            # Delete the user profile from MongoDB
            mongo_db.users.delete_one({'_id': ObjectId(user_id)})
            messages.success(request, 'Profile deleted successfully!')
            return redirect('logout')

    context = {
        'user_profile': user_profile,
    }
    return render(request, 'dashboard.html', context)