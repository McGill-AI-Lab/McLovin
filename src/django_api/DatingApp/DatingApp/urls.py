"""
URL configuration for DatingApp project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from home import views as home_views # since every app has a views they should be distinclty named
from databaseInteraction import views as login_views 

urlpatterns = [
    path('admin/', admin.site.urls),
    path('credits/', home_views.credits,name ="Hello World"),
    path('child/', home_views.child,name ="child"),
    path('base/', home_views.base,name ="base"),
    path('import/', home_views.module,name ="imported"),
    path("login/",login_views.login_view),
    path("logout/",login_views.logout_view),
    path("signup/",login_views.signup_view),
]

