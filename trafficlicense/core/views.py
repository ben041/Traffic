from django.shortcuts import render
from django.views.generic import ListView, DetailView, CreateView
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Vehicle, PlateDetection
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import os
from django.db.models import Q

from django.utils import timezone

class VehicleListView(LoginRequiredMixin, ListView):
    model = Vehicle
    template_name = 'vehicles/vehicle_list.html'
    context_object_name = 'vehicles'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Vehicle.objects.all()
        search = self.request.GET.get('search')
        vehicle_type = self.request.GET.get('vehicle_type')
        
        if search:
            queryset = queryset.filter(
                Q(plate_number__icontains=search) |
                Q(owner_name__icontains=search)
            )
        if vehicle_type:
            queryset = queryset.filter(vehicle_type=vehicle_type)
            
        return queryset.order_by('-created_at')
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['today'] = timezone.now().date()
        return context

class VehicleDetailView(LoginRequiredMixin, DetailView):
    model = Vehicle
    template_name = 'vehicles/vehicle_detail.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['today'] = timezone.now().date()
        return context
    

# views.py
from django.shortcuts import render
from django.views.generic import ListView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from .models import Vehicle, PlateDetection
import cv2
import numpy as np
import pytesseract
from PIL import Image
import io
import os

# Set Tesseract path - Update this path to match your installation
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def process_plate_detection(request):
    if request.method == 'POST' and (request.FILES.get('image') or request.FILES.get('video')):
        # Function to detect and process license plate from image
        def detect_plate(image):
            try:
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Apply filters and thresholding
                blur = cv2.GaussianBlur(gray, (5,5), 0)
                thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY_INV, 11, 2)
                
                # Find contours
                contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                
                # Filter contours based on area and aspect ratio
                possible_plates = []
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    if area > 1000:
                        x,y,w,h = cv2.boundingRect(cnt)
                        aspect_ratio = w/h
                        if 2.0 <= aspect_ratio <= 5.5:
                            possible_plates.append((x,y,w,h))
                
                results = []
                for x,y,w,h in possible_plates:
                    plate_img = gray[y:y+h, x:x+w]
                    
                    # Convert to PIL Image for better OCR
                    pil_image = Image.fromarray(plate_img)
                    
                    # Use pytesseract with custom configuration
                    text = pytesseract.image_to_string(
                        pil_image,
                        config='--psm 7 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                    )
                    
                    # Clean the text
                    text = ''.join(e for e in text if e.isalnum())
                    if len(text) > 4:  # Minimum plate length
                        results.append((text, 0.8))  # Confidence hardcoded for example
                
                return results
            except Exception as e:
                print(f"Error in detect_plate: {str(e)}")
                return []

        try:
            if request.FILES.get('image'):
                # Process single image
                image_file = request.FILES['image']
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), 
                                   cv2.IMREAD_COLOR)
                results = detect_plate(image)
                
            elif request.FILES.get('video'):
                # Process video
                video_file = request.FILES['video']
                temp_path = 'temp_video.mp4'
                with open(temp_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)
                
                cap = cv2.VideoCapture(temp_path)
                results = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_results = detect_plate(frame)
                    results.extend(frame_results)
                cap.release()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
            # Process results
            detections = []
            for plate_text, confidence in results:
                try:
                    vehicle = Vehicle.objects.get(plate_number=plate_text)
                    detection = PlateDetection.objects.create(
                        vehicle=vehicle,
                        detected_plate=plate_text,
                        confidence=confidence,
                        image=request.FILES.get('image', None),
                        video_file=request.FILES.get('video', None)
                    )
                    detections.append({
                        'plate': plate_text,
                        'owner': vehicle.owner_name,
                        'vehicle_type': vehicle.vehicle_type,
                        'confidence': confidence,
                        'make': vehicle.make,
                        'model': vehicle.model
                    })
                except Vehicle.DoesNotExist:
                    detections.append({
                        'plate': plate_text,
                        'error': 'Vehicle not found in database',
                        'confidence': confidence
                    })
            
            return render(request, 'detection_results.html', {'detections': detections})
        except Exception as e:
            return render(request, 'upload_form.html', {'error': f"Error processing image/video: {str(e)}"})
    
    return render(request, 'upload_form.html')