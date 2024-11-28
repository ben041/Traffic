# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.VehicleListView.as_view(), name='vehicle-list'),
#     path('vehicle/<int:pk>/', views.VehicleDetailView.as_view(), name='vehicle-detail'),
#     path('detect/', views.process_plate_detection, name='detect-plate'),
# ]


from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('Area list/', views.area_list, name='area_list'),

    path('video_feed/<int:area_id>/', views.video_feed, name='video_feed'),
    path('toggle_video_source/<int:area_id>/', views.toggle_video_source, name='toggle_video_source'),

    path('upload/', views.upload_video, name='upload_video'),
    path('suspected/', views.suspected_vehicles, name='suspected_vehicles'),
    path('all_vehicles/', views.all_vehicles, name='all_vehicles'),
    path('vehicle/<str:plate_number>/', views.vehicle_details, name='vehicle_details'),
]
