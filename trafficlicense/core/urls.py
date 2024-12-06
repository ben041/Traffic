# from django.urls import path
# from . import views

# urlpatterns = [
#     path('', views.VehicleListView.as_view(), name='vehicle-list'),
#     path('vehicle/<str:plate_number>/', views.VehicleDetailView.as_view(), name='vehicle-detail'),
#     path('detect/', views.process_plate_detection, name='detect-plate'),
# ]


from django.urls import path
from . import views

urlpatterns = [
    path('Home', views.home, name='home'),
    path('Area list/', views.area_list, name='area_list'),

    path('video_feed/<int:area_id>/', views.video_feed, name='video_feed'),

    path('video_feed/', views.video_feed1, name='video_feed1'),

    path('toggle_video_source/<int:area_id>/', views.toggle_video_source, name='toggle_video_source'),
    path('start_plate_detection/<int:area_id>/', views.start_plate_detection, name='start_plate_detection'),

    # path('upload/', views.upload_video, name='upload_video'),
    path('suspected/', views.suspected_vehicles, name='suspected_vehicles'),
    path('all_vehicles/', views.all_vehicles, name='all_vehicles'),
    path('vehicle/<str:plate_number>/', views.vehicle_details, name='vehicle_details'),

    path('area/add/', views.add_edit_area, name='add_area'),
    path('area/edit/<int:area_id>/', views.add_edit_area, name='edit_area'),
    path('vehicle/add-new/', views.add_edit_vehicle, name='add_vehicle'),
    path('vehicle/edit/<str:plate_number>/', views.add_edit_vehicle, name='edit_vehicle'),
    path('suspect-vehicle/add/', views.add_edit_suspect_vehicle, name='add_suspect_vehicle'),
    path('suspect-vehicle/edit/<str:plate_number>/', views.add_edit_suspect_vehicle, name='edit_suspect_vehicle'),

    path('area/delete/<str:plate_number>/', views.delete_area, name='delete_area'),
    path('vehicle/delete/<str:plate_number>/', views.delete_vehicle, name='delete_vehicle'),
    path('suspect-vehicle/delete/<str:plate_number>/', views.delete_suspect_vehicle, name='delete_suspect_vehicle'),



    path('signup/', views.signup_view, name='signup'),
    path('', views.signin_view, name='signin'),
    path('logout/', views.logout_view, name='logout'),
]
