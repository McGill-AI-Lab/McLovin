# views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from .mongo_client import get_mongodb_connection  # Import the MongoDB connection helper
from bson.objectid import ObjectId  # For handling ObjectId if needed


def user_dashboard(request, user_id):
    db = get_mongodb_connection()
    user_collection = db['users']  # Replace 'users' with your actual MongoDB collection name

    # Retrieve the user profile from the MongoDB collection
    user_profile = user_collection.find_one({'user_id': user_id})

    if not user_profile:
        messages.error(request, 'User profile not found.')
        return redirect('home')

    if request.method == 'POST':
        if 'edit' in request.POST:
            # Update the fields provided by the POST request
            updated_data = {
                'name': request.POST.get('name', user_profile.get('name', '')),
                'age': int(request.POST.get('age', user_profile.get('age', 0))),
                'grade': request.POST.get('grade', user_profile.get('grade', '')),
                'ethnicity': request.POST.getlist('ethnicity') or user_profile.get('ethnicity', []),
                'faculty': request.POST.get('faculty', user_profile.get('faculty', '')),
                'major': request.POST.getlist('major') or user_profile.get('major', []),
                'bio': request.POST.get('bio', user_profile.get('bio', '')),
                'preferences': request.POST.get('preferences', user_profile.get('preferences', '')),
            }

            # Update the user profile in MongoDB
            user_collection.update_one({'user_id': user_id}, {'$set': updated_data})
            messages.success(request, 'Profile updated successfully!')
            return redirect('user_dashboard', user_id=user_id)

        elif 'delete' in request.POST:
            # Delete the user profile from MongoDB
            user_collection.delete_one({'user_id': user_id})
            messages.success(request, 'Profile deleted successfully!')
            return redirect('home')

    context = {
        'user_profile': user_profile,
    }
    return render(request, 'dashboard.html', context)
