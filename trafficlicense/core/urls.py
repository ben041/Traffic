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
    path('', views.area_list, name='area_list'),
    path('video_feed/<int:area_id>/', views.video_feed, name='video_feed'),
    path('upload/', views.upload_video, name='upload_video'),
    path('suspected/', views.suspected_vehicles, name='suspected_vehicles'),
]
