from django import forms 

class ForecastForm(forms.Form):
    ticker = forms.CharField(max_length = 10, initial="AAPL")
    days = forms.IntegerField(min_value=1, max_value=30, initial=7)
    
    
    
