# DatingApp/context_processors.py

from django.conf import settings

def user_logged_in(request):
    logged_in = "user_id" in request.session
    user_id = -1
    if logged_in:
        user_id = request.session.get("user_id")
    return {
        'loggedIn': logged_in,
        'options': request.session.keys(),
        'user_id': user_id,
    }
