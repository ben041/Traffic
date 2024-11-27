from django.db import models
from geopy.geocoders import Nominatim

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
    video = models.FileField(upload_to='videos/', null=True, blank=True)
    camera_feed_url = models.URLField(null=True, blank=True)

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
    is_suspected = models.BooleanField(default=False)  # Mark as suspected vehicle
    reason_suspected = models.TextField(null=True, blank=True)  # Reason for suspicion
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.plate_number} - {self.owner_name}"

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
