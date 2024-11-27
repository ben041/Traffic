from django import forms
from .models import Area

class VideoUploadForm(forms.Form):
    area = forms.ModelChoiceField(queryset=Area.objects.all(), label="Select Area")
    video = forms.FileField(label="Upload Video")
