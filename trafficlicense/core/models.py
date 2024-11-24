from django.db import models
from django.contrib.auth.models import User

class Vehicle(models.Model):
    plate_number = models.CharField(max_length=20, unique=True)
    owner_name = models.CharField(max_length=100)
    owner_phone = models.CharField(max_length=20)
    owner_address = models.TextField()
    vehicle_type = models.CharField(max_length=50)
    make = models.CharField(max_length=50)
    model = models.CharField(max_length=50)
    year = models.IntegerField()
    color = models.CharField(max_length=30)
    registration_date = models.DateField()
    insurance_expiry = models.DateField()
    last_inspection_date = models.DateField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.plate_number} - {self.owner_name}"

class PlateDetection(models.Model):
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, null=True)
    detected_plate = models.CharField(max_length=20)
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='detections/')
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    
    def __str__(self):
        return f"Detection: {self.detected_plate} at {self.timestamp}"
