from djongo import models
from pymongo import MongoClient
from django.contrib.auth.models import AbstractBaseUser, BaseUserManager
from functools import wraps
from django.http import JsonResponse
import json
from django.shortcuts import redirect


def login_required(view_func):
    @wraps(view_func)
    def wrapper(request, *args, **kwargs):
        # Check if the user is authenticated
        if not request.session.get("user_id"):
            #return JsonResponse({"error": "Authentication required"}, status=401)
            # redrect to login and put the next as the current endpoint
            #TODO redirect to login and add ?next=<origin endpoint>
            return redirect(request,"login")#JsonResponse({"error":"please authenticate"}, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper