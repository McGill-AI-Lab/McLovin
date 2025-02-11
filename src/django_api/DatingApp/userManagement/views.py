
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from .user import login_required
from pymongo import MongoClient
from django.http import JsonResponse
from django.shortcuts import render, redirect
import json
import bcrypt
from .user import User

from dotenv import dotenv_values, find_dotenv,load_dotenv

from django.shortcuts import render, get_object_or_404, redirect
from django.contrib import messages
from bson import ObjectId
import sys
sys.path.append('../../')
from core.profile import UserProfile, Grade,Ethnicity,Faculty

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

            # Saves the user with credentials and default values here
            user = {
                "username": firstName+" "+lastName,
                "email": email,
                "password": hashed_password.decode(),  # Store the hashed password as a string
                'age': 0,
                'grade': 0,
                'ethnicity': [],# default list
                'faculty': "Faculty",
                'major': [], # default list
                'bio': "This is my bio !",
                'preferences': "preference",
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

def get_paremeters():
    dic = {}
    for enum in Grade, Faculty, Ethnicity:
        dic[str(enum.__name__).lower()] = [e.name for e in enum]
    initials = User().to_dict_with_defaults()
    for key,value in initials.items():
        if key in dic:
            initials[key] = dic[key]

    return initials

@login_required
def user_dashboard(request, user_id):
    user_profile = users_collection.find_one({"_id": ObjectId(user_id)})

    if not user_profile:
        messages.error(request, 'User profile not found.')
        return redirect('home')

    # Get default values from the User class
    default_values = User().to_dict_with_defaults()
    default_values.pop("user_id", None) # the user id is set on the server side
    default_values.pop("cluser_id", None)  # the user id is set on the server side

    if request.method == 'POST':
        if 'edit' in request.POST:
            # Merge user input with existing profile, ensuring defaults for missing fields
            updated_data = {
                field: request.POST.get(field, user_profile.get(field, default_values[field]))
                for field in default_values
            }

            dic = {}
            for enum in Grade, Faculty, Ethnicity:
                dic[str(enum.__name__).lower()] = [{e.name:e} for e in enum]

            # Update the user profile in MongoDB
            mongo_db.users.update_one({'_id': ObjectId(user_id)}, {'$set': updated_data})
            messages.success(request, 'Profile updated successfully!')

            # now perform conversion to inject into the pinecone database
            def map(grade_list, grade_value):
                for grade_dict in grade_list:
                    if grade_value in grade_dict:
                        return grade_dict[grade_value]
                return None  # Return None if the grade_value is not found

            for to_translate in 'grade','faculty','ethnicity':
                #print(map(dic[to_translate], updated_data[to_translate]).value) would print smt like Grade.U0
                updated_data[to_translate] = map(dic[to_translate], updated_data[to_translate])


            print(updated_data)

            # Now check if the User has complete data, and if so perform embedding on his profile
            incomplete = False
            for field,value in updated_data.items():
                if not value:
                    #messages.error(request, f'Please fill in the {field} field.')
                    #return redirect('user_dashboard', user_id=user_id)
                    incomplete = True

            if not incomplete:

                data = User().get_user_by_id(user_id)

                # DEBUGGING
                data.pop("_id", None) # remove the id from the data
                data.pop("username", None)
                data.pop("email", None)
                data.pop("password", None)
                data.pop("cluster_id", None)

                data["user_id"] =4948485875 # insert the user_id at the beginning of the list

                print(data)
                print("THIS WAS DATA")
                user_collection = UserProfile(**data)

                print(user_collection.tostring())
                # create a UserProfile instance to perform the embedding
                User().embed(user_collection)

                print("User has been successfully embedded !")

            return redirect('user_dashboard', user_id=user_id)

        elif 'delete' in request.POST:
            # Delete the user profile from MongoDB
            mongo_db.users.delete_one({'_id': ObjectId(user_id)})
            messages.success(request, 'Profile deleted successfully!')
            return redirect('logout')

    # Merge stored user profile with default values

    merged_profile = {field: user_profile.get(field, default_values[field]) for field in default_values}

    merged_profile["major"] = []

    context = {
        'user_profile': merged_profile,
        'default_info':get_paremeters(),
    }
    return render(request, 'dashboard.html', context)
