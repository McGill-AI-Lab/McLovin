from django.shortcuts import render
from django.http import HttpResponse 
from django.http import JsonResponse 
from django.template import Template, Context
#from django.contrib.auth.decorators import login_required
import sys
from userManagement.models import login_required

def credits(request):
    sample_dict = {
    "name": "John Doe",
    "age": 30,
    "isEmployed": True,
    "skills": ["Python", "JavaScript", "SQL"],
    "address": {
        "street": "123 Main St",
        "city": "Anytown",
        "state": "CA",
        "zipCode": "12345"
    },
    "projects": [
        {
            "title": "Website Redesign",
            "year": 2022,
            "role": "Frontend Developer"
        },
        {
            "title": "Data Analysis Tool",
            "year": 2023,
            "role": "Data Scientist"
        }
    ]
}
    template = Template("home.html")
    context = Context()
    return render(request,"home.html")
    #JsonResponse(data=sample_dict,safe=True)
    #HttpResponse(content, content_type=" text/html") #https://stackoverflow.com/questions/23714383/what-are-all-the-possible-values-for-http-content-type-header


def child(request):
    
    return render(request,"child.html")


def base(request):
    
    return render(request,"base.html")

def module(request):
    
    return render(request,"imported.html")

def home(request):
    # check if user is logged in, and add it to the context variables

    # retrieve the user instance of the logged in user to display name etc..
    loggedIn = False;
    if "user_id" in request.session.keys():
        loggedIn = True;
    context = {
        "loggedIn" : loggedIn,
        "options" : request.session.keys(),
    }

    return render(request,"home.html",context)

def privacy_policy(request):
    return render(request,"privacy_policy.html")

def terms_of_service(request):
    return render(request,"terms_of_service.html")
