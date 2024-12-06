# from django.shortcuts import render
# from django.views.generic import ListView, DetailView, CreateView
# from django.contrib.auth.mixins import LoginRequiredMixin
# from .models import Vehicle, PlateDetection
# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image
# import io
# import os
# from django.db.models import Q

# from django.utils import timezone

# class VehicleListView(LoginRequiredMixin, ListView):
#     model = Vehicle
#     template_name = 'vehicles/vehicle_list.html'
#     context_object_name = 'vehicles'
#     paginate_by = 10
    
#     def get_queryset(self):
#         queryset = Vehicle.objects.all()
#         search = self.request.GET.get('search')
#         vehicle_type = self.request.GET.get('vehicle_type')
        
#         if search:
#             queryset = queryset.filter(
#                 Q(plate_number__icontains=search) |
#                 Q(owner_name__icontains=search)
#             )
#         if vehicle_type:
#             queryset = queryset.filter(vehicle_type=vehicle_type)
            
#         return queryset.order_by('-created_at')
    
#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['today'] = timezone.now().date()
#         return context

# class VehicleDetailView(LoginRequiredMixin, DetailView):
#     model = Vehicle
#     template_name = 'vehicles/vehicle_detail.html'
    
#     def get_context_data(self, **kwargs):
#         context = super().get_context_data(**kwargs)
#         context['today'] = timezone.now().date()
#         return context
    

# # views.py
# from django.shortcuts import render
# from django.views.generic import ListView, DetailView
# from django.contrib.auth.mixins import LoginRequiredMixin
# from .models import Vehicle, PlateDetection
# import cv2
# import numpy as np
# import pytesseract
# from PIL import Image
# import io
# import os

# # Set Tesseract path - Update this path to match your installation
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def process_plate_detection(request):
#     if request.method == 'POST' and (request.FILES.get('image') or request.FILES.get('video')):
#         # Function to detect and process license plate from image
#         def detect_plate(image):
#             try:
#                 # Convert to grayscale
#                 gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
#                 # Apply filters and thresholding
#                 blur = cv2.GaussianBlur(gray, (5,5), 0)
#                 thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                             cv2.THRESH_BINARY_INV, 11, 2)
                
#                 # Find contours
#                 contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
#                 # Filter contours based on area and aspect ratio
#                 possible_plates = []
#                 for cnt in contours:
#                     area = cv2.contourArea(cnt)
#                     if area > 1000:
#                         x,y,w,h = cv2.boundingRect(cnt)
#                         aspect_ratio = w/h
#                         if 2.0 <= aspect_ratio <= 5.5:
#                             possible_plates.append((x,y,w,h))
                
#                 results = []
#                 for x,y,w,h in possible_plates:
#                     plate_img = gray[y:y+h, x:x+w]
                    
#                     # Convert to PIL Image for better OCR
#                     pil_image = Image.fromarray(plate_img)
                    
#                     # Use pytesseract with custom configuration
#                     text = pytesseract.image_to_string(
#                         pil_image,
#                         config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
#                     )
                    
#                     # Clean the text
#                     text = ''.join(e for e in text if e.isalnum())
#                     if len(text) > 4:  # Minimum plate length
#                         results.append((text, 0.8))  # Confidence hardcoded for example
                
#                 return results
#             except Exception as e:
#                 print(f"Error in detect_plate: {str(e)}")
#                 return []

#         try:
#             if request.FILES.get('image'):
#                 # Process single image
#                 image_file = request.FILES['image']
#                 image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 
#                                    cv2.IMREAD_COLOR)
#                 results = detect_plate(image)
                
#             elif request.FILES.get('video'):
#                 # Process video
#                 video_file = request.FILES['video']
#                 temp_path = 'temp_video.mp4'
#                 with open(temp_path, 'wb+') as destination:
#                     for chunk in video_file.chunks():
#                         destination.write(chunk)
                
#                 cap = cv2.VideoCapture(temp_path)
#                 results = []
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#                     frame_results = detect_plate(frame)
#                     results.extend(frame_results)
#                 cap.release()
#                 if os.path.exists(temp_path):
#                     os.remove(temp_path)
            
#             # Process results
#             detections = []
#             for plate_text, confidence in results:
#                 try:
#                     vehicle = Vehicle.objects.get(plate_number=plate_text)
#                     detection = PlateDetection.objects.create(
#                         vehicle=vehicle,
#                         detected_plate=plate_text,
#                         confidence=confidence,
#                         image=request.FILES.get('image', None),
#                         video_file=request.FILES.get('video', None)
#                     )
#                     detections.append({
#                         'plate': plate_text,
#                         'owner': vehicle.owner_name,
#                         'vehicle_type': vehicle.vehicle_type,
#                         'confidence': confidence,
#                         'make': vehicle.make,
#                         'model': vehicle.model
#                     })
#                 except Vehicle.DoesNotExist:
#                     detections.append({
#                         'plate': plate_text,
#                         'error': 'Vehicle not found in database',
#                         'confidence': confidence
#                     })
            
#             return render(request, 'detection_results.html', {'detections': detections})
#         except Exception as e:
#             return render(request, 'upload_form.html', {'error': f"Error processing image/video: {str(e)}"})
    
#     return render(request, 'upload_form.html')

from django.shortcuts import render, get_object_or_404
from .models import Area, Vehicle, PlateDetection
from .forms import VideoUploadForm
from django.http import JsonResponse
import cv2
import pytesseract
import os

# Configure the path to Tesseract OCR (adjust for your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def home(request):
    total_vehicles = Vehicle.objects.count()
    total_areas = Area.objects.count()
    total_suspects = SuspectVehicle.objects.count()
    vehicles = Vehicle.objects.all()

    context = {
        'total_vehicles': total_vehicles,
        'total_areas': total_areas,
        'total_suspects': total_suspects,
        'vehicles': vehicles,
    }
    return render(request, 'dashboard.html', context)

def area_list(request):
    """Display all camera installation areas."""
    areas = Area.objects.all()
    return render(request, 'area_list.html', {'areas': areas})


from django.shortcuts import render, get_object_or_404
from .models import Vehicle, SuspectVehicle

def suspected_vehicles(request):
    """Display all suspected vehicles."""
    suspected_list = SuspectVehicle.objects.select_related('vehicle').all()
    return render(request, 'suspected_vehicles.html', {'suspected_vehicles': suspected_list})

def all_vehicles(request):
    """Display all registered vehicles."""
    vehicles = Vehicle.objects.all()
    return render(request, 'all_vehicles.html', {'vehicles': vehicles})

def vehicle_details(request, plate_number):
    """Display details of a specific vehicle."""
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    suspect_details = SuspectVehicle.objects.filter(vehicle=vehicle).first()  # Get suspect details or None
    return render(request, 'vehicle_details.html', {
        'vehicle': vehicle,
        'suspect_details': suspect_details or SuspectVehicle(vehicle=vehicle)  # Use a default instance if None
    })


# def upload_video(request):
#     """Handle video upload and process plate detection."""
#     if request.method == 'POST':
#         form = VideoUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             video = form.cleaned_data['video']
#             area_id = form.cleaned_data['area']
#             area = Area.objects.get(id=area_id)
            
#             # Process the video file
#             process_video(video, area)
            
#             return JsonResponse({'status': 'success', 'message': 'Video processed successfully'})
#     else:
#         form = VideoUploadForm()
#     return render(request, 'upload_video.html', {'form': form})



# from django.shortcuts import render, get_object_or_404
# import cv2
# import pytesseract
# from django.http import JsonResponse
# from django.shortcuts import render, get_object_or_404
# import cv2
# import pytesseract
# from .models import Area, Vehicle, PlateDetection, SuspectVehicle

# def ajax_detect_plates(request, area_id):
#     """Detect number plates from video and return results as JSON."""
#     area = get_object_or_404(Area, id=area_id)
#     video_path = area.video.path if area.use_video_file else area.video_url

#     if not video_path:
#         return JsonResponse({'error': 'No video file or URL provided for this area.'}, status=400)

#     cap = cv2.VideoCapture(video_path)
#     plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
#     detected_plates = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))

#         for (x, y, w, h) in plates:
#             plate_img = frame[y:y + h, x:x + w]
#             plate_text = pytesseract.image_to_string(plate_img, config='--psm 8').strip().replace(' ', '')

#             # Check if the plate exists in the database
#             try:
#                 vehicle = Vehicle.objects.get(plate_number=plate_text)
#                 suspect_vehicle = SuspectVehicle.objects.filter(vehicle=vehicle, is_active=True).first()
#                 classification = "Suspect" if suspect_vehicle else "Not Suspect"

#                 # Save detection
#                 PlateDetection.objects.create(
#                     vehicle=vehicle,
#                     detected_plate=plate_text,
#                     confidence=0.95,  # Example confidence
#                     area=area
#                 )

#                 detected_plates.append({
#                     'plate': plate_text,
#                     'classification': classification,
#                     'vehicle_owner': vehicle.owner_name,
#                     'vehicle_details': f"{vehicle.make} {vehicle.model} ({vehicle.color})"
#                 })
#             except Vehicle.DoesNotExist:
#                 detected_plates.append({
#                     'plate': plate_text,
#                     'classification': "Unknown",
#                     'vehicle_owner': "N/A",
#                     'vehicle_details': "N/A"
#                 })

#     cap.release()
#     return JsonResponse({'plates': detected_plates}, safe=False)


# def process_video_and_display(request, area_id):
#     area = get_object_or_404(Area, id=area_id)
#     if not area.video:
#         return render(request, 'video_feed.html', {'error': 'No video file uploaded for this area.'})

#     # Detect plates and classify
#     detected_plates = detect_and_classify_plates(area.video, area)

#     context = {
#         'area': area,
#         'detected_plates': detected_plates,
#     }
#     return render(request, 'video_feed.html', context)

# from django.shortcuts import render, get_object_or_404, redirect
# from django.http import JsonResponse
# import cv2
# import pytesseract
# from .models import Area, Vehicle, PlateDetection, SuspectVehicle
# from asgiref.sync import async_to_sync
# from channels.layers import get_channel_layer

# channel_layer = get_channel_layer()  # For broadcasting updates via WebSocket


# def detect_and_classify_plates(video, area):
#     """Detect and classify number plates from a video."""
#     video_path = video.path  # Ensure this is a valid path
#     cap = cv2.VideoCapture(video_path)
#     plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
#     detected_plates = []  # Store plate details for display

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))

#         for (x, y, w, h) in plates:
#             plate_img = frame[y:y + h, x:x + w]
#             plate_text = pytesseract.image_to_string(plate_img, config='--psm 8').strip().replace(' ', '')

#             # Check if the plate exists in the Vehicle model
#             try:
#                 vehicle = Vehicle.objects.get(plate_number=plate_text)
#                 suspect_vehicle = SuspectVehicle.objects.filter(vehicle=vehicle, is_active=True).first()
#                 classification = "Suspect" if suspect_vehicle else "Not Suspect"

#                 # Save detection
#                 PlateDetection.objects.create(
#                     vehicle=vehicle,
#                     detected_plate=plate_text,
#                     confidence=0.95,  # Example confidence
#                     image=None,  # Add image saving logic if required
#                     video_file=video_path,
#                     area=area
#                 )

#                 detection_result = {
#                     'plate': plate_text,
#                     'classification': classification,
#                     'vehicle': {
#                         'owner_name': vehicle.owner_name,
#                         'make': vehicle.make,
#                         'model': vehicle.model,
#                     },
#                 }

#             except Vehicle.DoesNotExist:
#                 detection_result = {
#                     'plate': plate_text,
#                     'classification': "Unknown",
#                     'vehicle': None,
#                 }

#             detected_plates.append(detection_result)

#             # Send updates via WebSocket
#             async_to_sync(channel_layer.group_send)(
#                 f"area_{area.id}",
#                 {"type": "plate.detected", "plate_data": detection_result}
#             )

#     cap.release()
#     return detected_plates


# def video_feed(request, area_id):
#     area = get_object_or_404(Area, id=area_id)
#     context = {
#         'area': area,
#     }
#     return render(request, 'video_feed.html', context)


# def start_plate_detection(request, area_id):
#     """Start number plate detection and return detected plates."""
#     area = get_object_or_404(Area, id=area_id)
#     if not area.video and not area.video_url:
#         return JsonResponse({'error': 'No video source available for this area.'}, status=400)

#     if area.use_video_file and area.video:
#         detected_plates = detect_and_classify_plates(area.video, area)
#     elif area.video_url:
#         # Implement URL video handling if necessary
#         return JsonResponse({'error': 'Video URL detection is not yet implemented.'}, status=400)
#     else:
#         return JsonResponse({'error': 'No valid video source.'}, status=400)

#     return JsonResponse({'detected_plates': detected_plates})


# def toggle_video_source(request, area_id):
#     """Toggle between video file and video URL for the area."""
#     area = get_object_or_404(Area, id=area_id)
#     if request.method == 'POST':
#         # Toggle the use_video_file field
#         area.use_video_file = not area.use_video_file
#         area.save()
#     return redirect('video_feed', area_id=area.id)
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import cv2
import numpy as np
import pytesseract
from .models import Area, Vehicle, PlateDetection, SuspectVehicle
import logging
import requests
import tempfile
import os

# Configure logging
logger = logging.getLogger(__name__)

def download_video_from_url(video_url):
    """
    Download video from URL and save to a temporary file.
    Returns path to temporary file or None if download fails.
    """
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        for chunk in response.iter_content(chunk_size=8192):
            temp_file.write(chunk)
        temp_file.close()
        return temp_file.name
    except Exception as e:
        logger.error(f"Video download error: {e}")
        return None

def preprocess_plate_image(plate_img):
    """
    Preprocess the plate image to improve OCR accuracy.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

def detect_and_classify_plates(video_path, area):
    """
    Detect and classify number plates from a video.
    """
    cap = cv2.VideoCapture(video_path)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    detected_plates = []
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

    frame_count = 0
    detection_interval = 10

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % detection_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 25))

        for (x, y, w, h) in plates:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            plate_img = frame[y:y+h, x:x+w]
            preprocessed_plate = preprocess_plate_image(plate_img)

            try:
                plate_text = pytesseract.image_to_string(preprocessed_plate, config=custom_config).strip().replace(' ', '')
                plate_text = ''.join(char for char in plate_text if char.isalnum())
                if len(plate_text) < 4:
                    continue

                try:
                    vehicle = Vehicle.objects.get(plate_number=plate_text)
                    suspect_vehicle = SuspectVehicle.objects.filter(vehicle=vehicle, is_active=True).first()
                    classification = "Suspect" if suspect_vehicle else "Not Suspect"

                    PlateDetection.objects.create(
                        vehicle=vehicle,
                        detected_plate=plate_text,
                        confidence=0.85,
                        video_file=video_path,
                        area=area
                    )

                    detected_plates.append({
                        'plate': plate_text,
                        'classification': classification,
                        'vehicle': {
                            'owner_name': vehicle.owner_name,
                            'make': vehicle.make,
                            'model': vehicle.model,
                        },
                    })

                except Vehicle.DoesNotExist:
                    detected_plates.append({
                        'plate': plate_text,
                        'classification': "Unknown",
                        'vehicle': None,
                    })

            except Exception as e:
                logger.error(f"OCR Processing error: {e}")

    cap.release()
    return detected_plates

@login_required
def video_feed(request, area_id):
    """
    Render video feed page for a specific area.
    """
    area = get_object_or_404(Area, id=area_id)
    context = {'area': area}
    return render(request, 'video_feed.html', context)

@login_required
def start_plate_detection(request, area_id):
    """
    Start number plate detection and return detected plates.
    """
    area = get_object_or_404(Area, id=area_id)
    if not area.video and not area.video_url:
        return JsonResponse({'error': 'No video source available for this area.'}, status=400)

    try:
        if area.use_video_file and area.video:
            video_path = area.video.path
        elif area.video_url:
            video_path = download_video_from_url(area.video_url)
            if not video_path:
                return JsonResponse({'error': 'Failed to download video from URL.'}, status=400)
        else:
            return JsonResponse({'error': 'No valid video source.'}, status=400)

        detected_plates = detect_and_classify_plates(video_path, area)
        if area.video_url and 'temp' in video_path:
            os.unlink(video_path)

        return JsonResponse({'detected_plates': detected_plates})
    except Exception as e:
        logger.error(f"Plate detection error: {e}")
        return JsonResponse({'error': 'An unexpected error occurred during plate detection.'}, status=500)

@login_required
def toggle_video_source(request, area_id):
    """
    Toggle between video file and video URL for the area.
    """
    area = get_object_or_404(Area, id=area_id)
    if request.method == 'POST':
        area.use_video_file = not area.use_video_file
        area.save()
        messages.success(request, f"Video source switched to {'Video File' if area.use_video_file else 'Video URL'}")
    return redirect('video_feed', area_id=area.id)



from django.shortcuts import render, get_object_or_404, redirect
from .models import Area, Vehicle, SuspectVehicle
from .forms import AreaForm, VehicleForm, SuspectVehicleForm
from django.contrib import messages

# Add or Edit Area
def add_edit_area(request, area_id):  # Change parameter name for clarity
    area = get_object_or_404(Area, id=area_id)  # Use 'id' to fetch the area instance
    if request.method == 'POST':
        form = AreaForm(request.POST, request.FILES, instance=area)
        if form.is_valid():
            form.save()
            messages.success(request, 'Area saved successfully!')
            return redirect('area_list')  # Ensure 'area_list' is defined in your URLs
    else:
        form = AreaForm(instance=area)
    return render(request, 'add_edit_area.html', {'form': form, 'area': area})

# Add or Edit Vehicle
from django.shortcuts import get_object_or_404, redirect, render
from django.contrib import messages
from .models import Vehicle
from .forms import VehicleForm  # Make sure you have a form for Vehicle

def add_edit_vehicle(request, plate_number=None):
    if plate_number:
        # Editing an existing vehicle
        vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    else:
        # Adding a new vehicle
        vehicle = None

    if request.method == 'POST':
        form = VehicleForm(request.POST, instance=vehicle)
        if form.is_valid():
            form.save()
            if vehicle:
                messages.success(request, 'Vehicle updated successfully!')
            else:
                messages.success(request, 'Vehicle added successfully!')
            return redirect('vehicle_list')  # Replace with your vehicle list view name
    else:
        form = VehicleForm(instance=vehicle)

    return render(request, 'add_edit_vehicle.html', {'form': form, 'vehicle': vehicle})

# Add or Edit Suspect Vehicle
def add_edit_suspect_vehicle(request, plate_number):
    suspect_vehicle = SuspectVehicle.objects.filter(vehicle__plate_number=plate_number).first()
    if not suspect_vehicle:
        # Get or create the related vehicle
        vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
        suspect_vehicle = SuspectVehicle(vehicle=vehicle)

    if request.method == 'POST':
        form = SuspectVehicleForm(request.POST, instance=suspect_vehicle)
        if form.is_valid():
            form.save()
            messages.success(request, 'Suspect Vehicle saved successfully!')
            return redirect('suspected_vehicles')  # Replace with the name of the suspect vehicle list view
    else:
        form = SuspectVehicleForm(instance=suspect_vehicle)

    return render(request, 'add_edit_suspect_vehicle.html', {'form': form, 'suspect_vehicle': suspect_vehicle})

from django.shortcuts import get_object_or_404, redirect
from django.contrib import messages
from django.urls import reverse
from django.views.generic import DeleteView
from django.http import HttpResponseRedirect
from .models import Area, Vehicle, SuspectVehicle

# Area Delete View
def delete_area(request, plate_number):
    area = get_object_or_404(Area, plate_number=plate_number)
    if request.method == "POST":
        area.delete()
        messages.success(request, "Area deleted successfully!")
        return redirect('area_list')  # Replace 'area_list' with your area list view name
    return render(request, 'confirm_delete.html', {'object': area, 'type': 'Area'})

# Vehicle Delete View
def delete_vehicle(request, plate_number):
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    if request.method == "POST":
        vehicle.delete()
        messages.success(request, "Vehicle deleted successfully!")
        return redirect('all_vehicles')  # Replace 'vehicle_list' with your vehicle list view name
    return render(request, 'confirm_delete.html', {'object': vehicle, 'type': 'Vehicle'})

# SuspectVehicle Delete View
def delete_suspect_vehicle(request, plate_number):
    suspect_vehicle = get_object_or_404(SuspectVehicle, plate_number=plate_number)
    if request.method == "POST":
        suspect_vehicle.delete()
        messages.success(request, "Suspect Vehicle deleted successfully!")
        return redirect('suspected_vehicles')  # Replace 'suspect_vehicle_list' with your suspect vehicle list view name
    return render(request, 'confirm_delete.html', {'object': suspect_vehicle, 'type': 'Suspect Vehicle'})


from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.shortcuts import render, redirect
from django.contrib import messages

def signup_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            if User.objects.filter(username=username).exists():
                messages.error(request, "Username already exists.")
            elif User.objects.filter(email=email).exists():
                messages.error(request, "Email is already in use.")
            else:
                User.objects.create_user(username=username, email=email, password=password1)
                messages.success(request, "Account created successfully.")
                return redirect('home')
        else:
            messages.error(request, "Passwords do not match.")
    return render(request, 'signup.html')


def signin_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password) 
        if user is not None:
            login(request, user)
            messages.success(request, "Logged in successfully.")
            return redirect('home')  # Redirect to home/dashboard page
        else:
            messages.error(request, "Invalid email or password.")
    return render(request, 'signin.html')


def logout_view(request):
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('signin')













from django.http import StreamingHttpResponse
import cv2
import pytesseract

# Set up Tesseract path if necessary
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def generate_frames():
    # Load Haar cascade for license plates
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    cap = cv2.VideoCapture(0)  # Open webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect plates
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in plates:
            # Draw rectangle and extract text
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            plate_roi = frame[y:y + h, x:x + w]
            text = pytesseract.image_to_string(plate_roi, config='--psm 7')
            cv2.putText(frame, text.strip(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def video_feed1(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')






















