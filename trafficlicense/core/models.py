from django.db import models
from geopy.geocoders import Nominatim
from django.contrib.auth.models import User

class Area(models.Model):
    DISTRICT_CHOICES = [
        ('Balaka', 'Balaka'),
        ('Blantyre', 'Blantyre'),
        ('Chikwawa', 'Chikwawa'),
        ('Chiradzulu', 'Chiradzulu'),
        ('Chitipa', 'Chitipa'),
        ('Dedza', 'Dedza'),
        ('Dowa', 'Dowa'),
        ('Karonga', 'Karonga'),
        ('Kasungu', 'Kasungu'),
        ('Likoma', 'Likoma'),
        ('Lilongwe', 'Lilongwe'),
        ('Machinga', 'Machinga'),
        ('Mangochi', 'Mangochi'),
        ('Mchinji', 'Mchinji'),
        ('Mulanje', 'Mulanje'),
        ('Mwanza', 'Mwanza'),
        ('Mzimba', 'Mzimba'),
        ('Nkhata Bay', 'Nkhata Bay'),
        ('Nkhotakota', 'Nkhotakota'),
        ('Nsanje', 'Nsanje'),
        ('Ntcheu', 'Ntcheu'),
        ('Ntchisi', 'Ntchisi'),
        ('Phalombe', 'Phalombe'),
        ('Rumphi', 'Rumphi'),
        ('Salima', 'Salima'),
        ('Thyolo', 'Thyolo'),
        ('Zomba', 'Zomba'),
    ]
    name = models.CharField(max_length=100, choices=DISTRICT_CHOICES)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)
    description = models.TextField()
    video = models.FileField(upload_to='videos/', blank=True, null=True)
    video_url = models.URLField(blank=True, null=True)
    use_video_file = models.BooleanField(default=True)

    def save(self, *args, **kwargs):
        """Auto-fill latitude and longitude if not provided."""
        if not self.latitude or not self.longitude:
            geolocator = Nominatim(user_agent="geoapi")
            location = geolocator.geocode(self.name)
            if location:
                self.latitude = location.latitude
                self.longitude = location.longitude
        super().save(*args, **kwargs)

    def __str__(self):
        return self.name

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

class DetectedPlate(models.Model):
    plate = models.CharField(max_length=20, unique=True)
    classification = models.CharField(max_length=20, choices=[
        ('Suspect', 'Suspect'),
        ('Not Suspect', 'Not Suspect'),
        ('Unknown', 'Unknown'),
    ])
    vehicle = models.ForeignKey(Vehicle, on_delete=models.SET_NULL, null=True, blank=True)
    area_id = models.IntegerField()
    confidence = models.FloatField(default=0.0)  # Added confidence field

    def __str__(self):
        return f"{self.plate} ({self.confidence:.2f})"

class PlateDetection(models.Model):
    vehicle = models.ForeignKey(Vehicle, on_delete=models.CASCADE, null=True)
    detected_plate = models.CharField(max_length=20)
    confidence = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='detections/', null=True, blank=True)
    video_file = models.FileField(upload_to='videos/', null=True, blank=True)
    area = models.ForeignKey(Area, on_delete=models.SET_NULL, null=True)

    def __str__(self):
        return f"Detection: {self.detected_plate} at {self.timestamp}"

class SuspectVehicle(models.Model):
    vehicle = models.OneToOneField(Vehicle, on_delete=models.CASCADE, related_name='suspect_details')
    reported_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='reported_suspects')
    reason_suspected = models.TextField()
    crime_committed = models.CharField(max_length=255)
    crime_details = models.TextField(null=True, blank=True)
    reported_date = models.DateField(auto_now_add=True)
    police_station = models.CharField(max_length=100, null=True, blank=True)
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return f"Suspect Vehicle: {self.vehicle.plate_number} - {self.crime_committed}"