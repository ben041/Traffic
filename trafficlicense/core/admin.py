from django.contrib import admin
from .models import *

# Register your models here.
admin.site.register(Area)
admin.site.register(Vehicle)
admin.site.register(PlateDetection)
admin.site.register(SuspectVehicle)
admin.site.register(DetectedPlate)