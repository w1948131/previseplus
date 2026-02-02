from django.urls import path
from django.shortcuts import redirect
from .views import register, CustomLoginView, logout_view


urlpatterns = [
    path("register/", register, name="register"),
    path("login/", CustomLoginView.as_view(), name="login"),
    path("logout/", logout_view, name="logout"),
    path("accounts/profile/", lambda request: redirect("/dashboard/")),
]
