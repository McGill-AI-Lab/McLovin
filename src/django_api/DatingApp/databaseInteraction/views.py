from django.shortcuts import render


from django.contrib.auth import authenticate, login, logout
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import User
from .serializers import UserSerializer

@api_view(['GET', 'POST'])
@csrf_exempt
def login_view(request):
    if request.method == 'GET':
        return render(request,"login.html")
    elif request.method == 'POST':
        email = request.data.get('email')
        password = request.data.get('password')
        user = authenticate(request, email=email, password=password)
        if user is not None:
            login(request, user)
            serializer = UserSerializer(user)
            return Response(serializer.data)
        else:
            return Response({'error': 'Invalid email or password'}, status=status.HTTP_400_BAD_REQUEST)

@api_view(['GET', 'POST'])
@csrf_exempt
def logout_view(request):
    logout(request)
    return Response({'success': 'Logged out successfully'})

@api_view(['GET', 'POST'])
@csrf_exempt
def signup_view(request):
    if request.method == 'GET':
        return render(request,"signup.html")
    elif request.method == 'POST':
        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_405_BAD_REQUEST)
    
