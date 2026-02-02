from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView
from django.contrib.auth import logout

# Create your views here.
def register(request):
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()
    
    return render(request, "register.html", {"form": form})      


#login view
class CustomLoginView(LoginView):
    template_name = "login.html"  
    
    
    
#logout view
def logout_view(request):
    logout(request)
    return redirect("/")
        
        

        
        
        