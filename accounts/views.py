from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView
from django.contrib.auth import logout

# Create your views here.

#user registration
def register(request):
    if request.method == "POST":   
        form = UserCreationForm(request.POST) # registration form
        if form.is_valid():
            form.save() #creates user and saved in db
            return redirect("login") # user redirected to login
    else:
        form = UserCreationForm() 
    
    return render(request, "register.html", {"form": form})  #displays register page with form    


#login view
#custom login view
class CustomLoginView(LoginView):
    template_name = "login.html"  
    
    
    
#logout view
def logout_view(request):
    logout(request)
    return redirect("/") #redirected to homepage when logout
        
        

        
        
        