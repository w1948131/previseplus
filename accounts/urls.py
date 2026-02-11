from django.urls import path
from django.shortcuts import redirect
from .views import register, CustomLoginView, logout_view #authentication for class based views


urlpatterns = [
    path("register/", register, name="register"),
    path("login/", CustomLoginView.as_view(), name="login"), #renders custom login page, see views.py
    path("logout/", logout_view, name="logout"),
    path("accounts/profile/", lambda request: redirect("/dashboard/")), # meant for profile feature, not implemented yet 
]
