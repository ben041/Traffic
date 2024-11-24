from django.urls import path
from . import views

urlpatterns = [
    path('', views.VehicleListView.as_view(), name='vehicle-list'),
    path('vehicle/<int:pk>/', views.VehicleDetailView.as_view(), name='vehicle-detail'),
    path('detect/', views.process_plate_detection, name='detect-plate'),
]