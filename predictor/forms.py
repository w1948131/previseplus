from django import forms 

# stock prediction form 
class ForecastForm(forms.Form):
    ticker = forms.CharField(max_length = 10, initial="AAPL") # ticker symbol input
    days = forms.IntegerField(min_value=1, max_value=30, initial=7) # No of days forecast input
    
    
    
