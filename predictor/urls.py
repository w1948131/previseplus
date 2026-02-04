from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name = "home"),
    path("dashboard/", views.dashboard, name = "dashboard"),
    path("predict/<str:ticker_value>/<int:number_of_days>/", views.predict, name="predict"),
    path("tickers/", views.ticker, name="tickers"),
    path("search/", views.search, name="search")
    
]