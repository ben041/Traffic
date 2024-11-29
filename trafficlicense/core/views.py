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
    return render(request, 'dashboard.html')

def area_list(request):
    """Display all camera installation areas."""
    areas = Area.objects.all()
    return render(request, 'area_list.html', {'areas': areas})

from django.shortcuts import get_object_or_404, redirect, render
from .models import Area

# def video_feed(request, area_id):
#     area = get_object_or_404(Area, id=area_id)
#     context = {
#         'area': area
#     }
#     return render(request, 'video_feed.html', context)

def toggle_video_source(request, area_id):
    area = get_object_or_404(Area, id=area_id)
    if request.method == 'POST':
        # Toggle the use_video_file field
        area.use_video_file = not area.use_video_file
        area.save()
    return redirect('video_feed', area_id=area.id)


# def process_video(video, area):
#     """Process uploaded video for plate detection."""
#     video_path = os.path.join('media', 'videos', video.name)
#     cap = cv2.VideoCapture(video_path)
#     plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))
        
#         for (x, y, w, h) in plates:
#             plate_img = frame[y:y + h, x:x + w]
#             plate_text = pytesseract.image_to_string(plate_img, config='--psm 8').strip().replace(' ', '')
            
#             # Check if the plate exists in the system
#             try:
#                 vehicle = Vehicle.objects.get(plate_number=plate_text)
#                 PlateDetection.objects.create(
#                     vehicle=vehicle,
#                     detected_plate=plate_text,
#                     confidence=0.95,
#                     image='detections/example.jpg',
#                     video_file=video_path
#                 )
#             except Vehicle.DoesNotExist:
#                 continue

#     cap.release()

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
    suspected_details = SuspectVehicle.objects.filter(vehicle=vehicle).first()  # Check if the vehicle is suspected
    return render(request, 'vehicle_details.html', {
        'vehicle': vehicle,
        'suspect_details': suspected_details
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



from django.shortcuts import render, get_object_or_404
import cv2
import pytesseract
from .models import Area, Vehicle, PlateDetection, SuspectVehicle

def detect_and_classify_plates(video, area):
    """Detect and classify number plates from a video."""
    video_path = video.path  # Ensure this is a valid path
    cap = cv2.VideoCapture(video_path)
    plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')
    detected_plates = []  # Store plate details for display

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(25, 25))

        for (x, y, w, h) in plates:
            plate_img = frame[y:y + h, x:x + w]
            plate_text = pytesseract.image_to_string(plate_img, config='--psm 8').strip().replace(' ', '')

            # Check if the plate exists in the Vehicle model
            try:
                vehicle = Vehicle.objects.get(plate_number=plate_text)
                suspect_vehicle = SuspectVehicle.objects.filter(vehicle=vehicle, is_active=True).first()
                classification = "Suspect" if suspect_vehicle else "Not Suspect"

                # Save detection
                PlateDetection.objects.create(
                    vehicle=vehicle,
                    detected_plate=plate_text,
                    confidence=0.95,  # Example confidence
                    image=None,  # Add image saving logic if required
                    video_file=video_path,
                    area=area
                )

                detected_plates.append({
                    'plate': plate_text,
                    'classification': classification,
                    'vehicle': vehicle,
                })

            except Vehicle.DoesNotExist:
                detected_plates.append({
                    'plate': plate_text,
                    'classification': "Unknown",
                    'vehicle': None,
                })

    cap.release()
    return detected_plates

def process_video_and_display(request, area_id):
    area = get_object_or_404(Area, id=area_id)
    if not area.video:
        return render(request, 'video_feed.html', {'error': 'No video file uploaded for this area.'})

    # Detect plates and classify
    detected_plates = detect_and_classify_plates(area.video, area)

    context = {
        'area': area,
        'detected_plates': detected_plates,
    }
    return render(request, 'video_feed.html', context)




from django.shortcuts import render, get_object_or_404, redirect
from .models import Area, Vehicle, SuspectVehicle
from .forms import AreaForm, VehicleForm, SuspectVehicleForm
from django.contrib import messages

# Add or Edit Area
def add_edit_area(request, pk=None):
    area = get_object_or_404(Area, pk=pk) if pk else None
    if request.method == 'POST':
        form = AreaForm(request.POST, request.FILES, instance=area)
        if form.is_valid():
            form.save()
            messages.success(request, 'Area saved successfully!')
            return redirect('area_list')  # Replace with the name of the area list view
    else:
        form = AreaForm(instance=area)
    return render(request, 'add_edit_area.html', {'form': form, 'area': area})

# Add or Edit Vehicle
def add_edit_vehicle(request, pk=None):
    vehicle = get_object_or_404(Vehicle, pk=pk) if pk else None
    if request.method == 'POST':
        form = VehicleForm(request.POST, instance=vehicle)
        if form.is_valid():
            form.save()
            messages.success(request, 'Vehicle saved successfully!')
            return redirect('vehicle_list')  # Replace with the name of the vehicle list view
    else:
        form = VehicleForm(instance=vehicle)
    return render(request, 'add_edit_vehicle.html', {'form': form, 'vehicle': vehicle})

# Add or Edit Suspect Vehicle
def add_edit_suspect_vehicle(request, pk=None):
    suspect_vehicle = get_object_or_404(SuspectVehicle, pk=pk) if pk else None
    if request.method == 'POST':
        form = SuspectVehicleForm(request.POST, instance=suspect_vehicle)
        if form.is_valid():
            form.save()
            messages.success(request, 'Suspect Vehicle saved successfully!')
            return redirect('suspect_vehicle_list')  # Replace with the name of the suspect vehicle list view
    else:
        form = SuspectVehicleForm(instance=suspect_vehicle)
    return render(request, 'add_edit_suspect_vehicle.html', {'form': form, 'suspect_vehicle': suspect_vehicle})
