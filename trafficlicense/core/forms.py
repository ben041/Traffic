from django import forms
from .models import Area

class VideoUploadForm(forms.Form):
    area = forms.ModelChoiceField(queryset=Area.objects.all(), label="Select Area")
    video = forms.FileField(label="Upload Video")

from django import forms
from .models import Area, Vehicle, SuspectVehicle

class AreaForm(forms.ModelForm):
    class Meta:
        model = Area
        fields = [
            'name', 'latitude', 'longitude', 'description', 
            'video', 'video_url', 'use_video_file'
        ]
        widgets = {
            'name': forms.Select(attrs={'class': 'form-select'}),
            'latitude': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Latitude'}),
            'longitude': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Longitude'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Enter Description'}),
            'video': forms.FileInput(attrs={'class': 'form-control'}),
            'video_url': forms.URLInput(attrs={'class': 'form-control', 'placeholder': 'Enter Video URL'}),
            'use_video_file': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }


class VehicleForm(forms.ModelForm):
    class Meta:
        model = Vehicle
        fields = [
            'plate_number', 'owner_name', 'owner_phone', 'owner_address', 
            'vehicle_type', 'make', 'model', 'year', 'color', 
            'registration_date', 'insurance_expiry', 'last_inspection_date'
        ]
        widgets = {
            'plate_number': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Plate Number'}),
            'owner_name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Owner Name'}),
            'owner_phone': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Owner Phone'}),
            'owner_address': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Enter Owner Address'}),
            'vehicle_type': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Vehicle Type'}),
            'make': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Make'}),
            'model': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Model'}),
            'year': forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter Year'}),
            'color': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Color'}),
            'registration_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'insurance_expiry': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
            'last_inspection_date': forms.DateInput(attrs={'class': 'form-control', 'type': 'date'}),
        }


class SuspectVehicleForm(forms.ModelForm):
    class Meta:
        model = SuspectVehicle
        fields = [
            'vehicle', 'reported_by', 'reason_suspected', 'crime_committed', 
            'crime_details', 'police_station', 'is_active'
        ]
        widgets = {
            'vehicle': forms.Select(attrs={'class': 'form-select'}),
            'reported_by': forms.Select(attrs={'class': 'form-select'}),
            'reason_suspected': forms.Textarea(attrs={'class': 'form-control', 'rows': 3, 'placeholder': 'Enter Reason for Suspicion'}),
            'crime_committed': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Crime Committed'}),
            'crime_details': forms.Textarea(attrs={'class': 'form-control', 'rows': 4, 'placeholder': 'Enter Crime Details'}),
            'police_station': forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Enter Police Station'}),
            'is_active': forms.CheckboxInput(attrs={'class': 'form-check-input'}),
        }
