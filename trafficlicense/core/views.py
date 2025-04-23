# views.py
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse, HttpResponseRedirect
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.utils import timezone
from django.db.models import Q
from .models import Area, Vehicle, PlateDetection, SuspectVehicle, DetectedPlate
from .forms import AreaForm, VehicleForm, SuspectVehicleForm, VideoUploadForm
import cv2
import numpy as np
import pytesseract
import requests
import tempfile
import os
import logging
from PIL import Image
from ultralytics import YOLO
import easyocr

# Configure logging
logger = logging.getLogger(__name__)

# Configure Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load YOLO model
try:
    yolo_model = YOLO('media/models/best.pt')  # Path to your trained YOLO model
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    yolo_model = None

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'], gpu=False)

# ------------------- General Views -------------------

def home(request):
    """Display the dashboard with statistics and vehicle list."""
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

def gis_view(request):
    """Render the GIS view with area detection data."""
    areas = Area.objects.filter(latitude__isnull=False, longitude__isnull=False)
    area_data = [
        {
            'name': area.name,
            'latitude': area.latitude,
            'longitude': area.longitude,
            'detections': PlateDetection.objects.filter(area=area).count()
        }
        for area in areas
    ]
    return render(request, 'gis.html', {'areas': area_data})

# ------------------- Authentication Views -------------------

def signup_view(request):
    """Handle user signup."""
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
    """Handle user login."""
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "Logged in successfully.")
            return redirect('home')
        else:
            messages.error(request, "Invalid email or password.")
    return render(request, 'signin.html')

def logout_view(request):
    """Handle user logout."""
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('signin')

# ------------------- Vehicle Views -------------------

@login_required
def vehicle_list(request):
    """Display a paginated list of vehicles with search and filter options."""
    queryset = Vehicle.objects.all()
    search = request.GET.get('search')
    vehicle_type = request.GET.get('vehicle_type')

    if search:
        queryset = queryset.filter(
            Q(plate_number__icontains=search) |
            Q(owner_name__icontains=search)
        )
    if vehicle_type:
        queryset = queryset.filter(vehicle_type=vehicle_type)

    vehicles = queryset.order_by('-created_at')
    
    # Simple pagination
    page = int(request.GET.get('page', 1))
    per_page = 10
    total = len(vehicles)
    start = (page - 1) * per_page
    end = start + per_page
    paginated_vehicles = vehicles[start:end]
    
    context = {
        'vehicles': paginated_vehicles,
        'today': timezone.now().date(),
        'page': page,
        'total_pages': (total + per_page - 1) // per_page,
    }
    return render(request, 'vehicles/vehicle_list.html', context)

@login_required
def vehicle_detail(request, plate_number):
    """Display details of a specific vehicle."""
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    context = {
        'vehicle': vehicle,
        'today': timezone.now().date(),
    }
    return render(request, 'vehicles/vehicle_detail.html', context)

def all_vehicles(request):
    """Display all registered vehicles."""
    vehicles = Vehicle.objects.all()
    return render(request, 'all_vehicles.html', {'vehicles': vehicles})

def vehicle_details(request, plate_number):
    """Display details of a specific vehicle with suspect status."""
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    suspect_details = SuspectVehicle.objects.filter(vehicle=vehicle).first()
    return render(request, 'vehicle_details.html', {
        'vehicle': vehicle,
        'suspect_details': suspect_details or SuspectVehicle(vehicle=vehicle)
    })

def add_edit_vehicle(request, plate_number=None):
    """Add or edit a vehicle."""
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number) if plate_number else None

    if request.method == 'POST':
        form = VehicleForm(request.POST, instance=vehicle)
        if form.is_valid():
            form.save()
            message = "Vehicle updated successfully!" if vehicle else "Vehicle added successfully!"
            messages.success(request, message)
            return redirect('vehicle_list')
    else:
        form = VehicleForm(instance=vehicle)

    return render(request, 'add_edit_vehicle.html', {'form': form, 'vehicle': vehicle})

def delete_vehicle(request, plate_number):
    """Delete a vehicle."""
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    if request.method == "POST":
        vehicle.delete()
        messages.success(request, "Vehicle deleted successfully!")
        return redirect('all_vehicles')
    return render(request, 'confirm_delete.html', {'object': vehicle, 'type': 'Vehicle'})

# ------------------- Suspect Vehicle Views -------------------

def suspected_vehicles(request):
    """Display all suspected vehicles."""
    suspected_list = SuspectVehicle.objects.select_related('vehicle').all()
    return render(request, 'suspected_vehicles.html', {'suspected_vehicles': suspected_list})

def add_edit_suspect_vehicle(request, plate_number):
    """Add or edit a suspect vehicle."""
    suspect_vehicle = SuspectVehicle.objects.filter(vehicle__plate_number=plate_number).first()
    if not suspect_vehicle:
        vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
        suspect_vehicle = SuspectVehicle(vehicle=vehicle)

    if request.method == 'POST':
        form = SuspectVehicleForm(request.POST, instance=suspect_vehicle)
        if form.is_valid():
            form.save()
            messages.success(request, "Suspect Vehicle saved successfully!")
            return redirect('suspected_vehicles')
    else:
        form = SuspectVehicleForm(instance=suspect_vehicle)

    return render(request, 'add_edit_suspect_vehicle.html', {'form': form, 'suspect_vehicle': suspect_vehicle})

def delete_suspect_vehicle(request, plate_number):
    """Delete a suspect vehicle."""
    suspect_vehicle = get_object_or_404(SuspectVehicle, plate_number=plate_number)
    if request.method == "POST":
        suspect_vehicle.delete()
        messages.success(request, "Suspect Vehicle deleted successfully!")
        return redirect('suspected_vehicles')
    return render(request, 'confirm_delete.html', {'object': suspect_vehicle, 'type': 'Suspect Vehicle'})

# ------------------- Area Views -------------------

def area_list(request):
    """Display all camera installation areas."""
    areas = Area.objects.all()
    return render(request, 'area_list.html', {'areas': areas})

def add_edit_area(request, area_id):
    """Add or edit an area."""
    area = get_object_or_404(Area, id=area_id)
    if request.method == 'POST':
        form = AreaForm(request.POST, request.FILES, instance=area)
        if form.is_valid():
            form.save()
            messages.success(request, 'Area saved successfully!')
            return redirect('area_list')
    else:
        form = AreaForm(instance=area)
    return render(request, 'add_edit_area.html', {'form': form, 'area': area})

def delete_area(request, area_id):
    """Delete an area."""
    area = get_object_or_404(Area, id=area_id)
    if request.method == "POST":
        area.delete()
        messages.success(request, "Area deleted successfully!")
        return redirect('area_list')
    return render(request, 'confirm_delete.html', {'object': area, 'type': 'Area'})

# ------------------- Detection Log View -------------------

@login_required
def detection_log(request):
    """Display a log of all plate detections."""
    detections = PlateDetection.objects.select_related('vehicle', 'area').order_by('-timestamp')
    area_id = request.GET.get('area_id')
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    plate_number = request.GET.get('plate_number')

    if area_id:
        detections = detections.filter(area_id=area_id)
    if start_date:
        detections = detections.filter(timestamp__gte=start_date)
    if end_date:
        detections = detections.filter(timestamp__lte=end_date)
    if plate_number:
        detections = detections.filter(detected_plate__icontains=plate_number)

    areas = Area.objects.all()
    return render(request, 'detection_log.html', {
        'detections': detections,
        'areas': areas
    })

# ------------------- Plate Detection Views -------------------

def process_plate_detection(request):
    """Handle image or video upload and process plate detection."""
    if request.method == 'POST' and (request.FILES.get('image') or request.FILES.get('video')):
        def detect_plate(image):
            """Detect license plates in an image using YOLO."""
            try:
                if not yolo_model:
                    raise Exception("YOLO model not loaded")
                
                # Convert image to RGB for YOLO
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = yolo_model.predict(image_rgb, device='cpu')
                
                possible_plates = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        possible_plates.append((x1, y1, x2-x1, y2-y1, confidence))
                
                results = []
                for x, y, w, h, confidence in possible_plates:
                    plate_img = image[y:y+h, x:x+w]
                    plate_img = preprocess_plate_image(plate_img)
                    
                    # Use EasyOCR for plate text extraction
                    ocr_results = easyocr_reader.readtext(plate_img, detail=1)
                    for (bbox, text, ocr_confidence) in ocr_results:
                        text = ''.join(e for e in text if e.isalnum())
                        if len(text) > 4:  # Minimum plate length
                            results.append((text, ocr_confidence * confidence))  # Combine confidences
                
                return results
            except Exception as e:
                logger.error(f"Error in detect_plate: {str(e)}")
                return []

        try:
            if request.FILES.get('image'):
                image_file = request.FILES['image']
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8),
                                     cv2.IMREAD_COLOR)
                results = detect_plate(image)
                
            elif request.FILES.get('video'):
                video_file = request.FILES['video']
                temp_path = 'temp_video.mp4'
                with open(temp_path, 'wb+') as destination:
                    for chunk in video_file.chunks():
                        destination.write(chunk)
                
                cap = cv2.VideoCapture(temp_path)
                results = []
                frame_count = 0
                process_interval = 5
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_count += 1
                    if frame_count % process_interval == 0:
                        frame_results = detect_plate(frame)
                        results.extend(frame_results)
                cap.release()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            
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
            logger.error(f"Error processing image/video: {str(e)}")
            return render(request, 'upload_form.html', {'error': f"Error processing image/video: {str(e)}"})
    
    return render(request, 'upload_form.html')

def download_video_from_url(video_url):
    """Download video from URL and save to a temporary file."""
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
    """Advanced preprocessing for license plate image."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return gray

def detect_and_classify_plates(video_path, area):
    """Detect and classify number plates from a video using YOLO."""
    if not yolo_model:
        logger.error("YOLO model not loaded")
        return []

    cap = cv2.VideoCapture(video_path)
    detected_plates = []
    frame_count = 0
    detection_interval = 5

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % detection_interval != 0:
            continue

        # YOLO detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = yolo_model.predict(rgb_frame, device='cpu')

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                
                # Draw rectangle for visualization
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Extract plate region for OCR
                plate_img = frame[y1:y2, x1:x2]
                preprocessed_plate = preprocess_plate_image(plate_img)
                
                # Use EasyOCR for plate text extraction
                ocr_results = easyocr_reader.readtext(preprocessed_plate, detail=1)
                for (bbox, plate_text, ocr_confidence) in ocr_results:
                    plate_text = ''.join(char for char in plate_text if char.isalnum())
                    if len(plate_text) < 4:
                        continue

                    try:
                        vehicle = Vehicle.objects.get(plate_number=plate_text)
                        suspect_vehicle = SuspectVehicle.objects.filter(vehicle=vehicle, is_active=True).first()
                        classification = "Suspect" if suspect_vehicle else "Not Suspect"

                        detection = PlateDetection.objects.create(
                            vehicle=vehicle,
                            detected_plate=plate_text,
                            confidence=ocr_confidence * confidence,  # Combine YOLO and OCR confidence
                            video_file=video_path,
                            area=area
                        )

                        detection_result = {
                            'plate': plate_text,
                            'classification': classification,
                            'vehicle': {
                                'owner_name': vehicle.owner_name,
                                'make': vehicle.make,
                                'model': vehicle.model,
                            },
                        }

                        detected_plates.append(detection_result)
                    except Vehicle.DoesNotExist:
                        detected_plates.append({
                            'plate': plate_text,
                            'classification': "Unknown",
                            'vehicle': None,
                        })

    cap.release()
    return detected_plates

@login_required
def video_feed(request, area_id):
    """Render video feed for a specific area."""
    area = get_object_or_404(Area, id=area_id)
    context = {
        'area': area,
    }
    return render(request, 'video_feed.html', context)

@login_required
def start_plate_detection(request, area_id):
    """Start number plate detection and return detected plates."""
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
    """Toggle between video file and video URL for the area."""
    area = get_object_or_404(Area, id=area_id)
    if request.method == 'POST':
        area.use_video_file = not area.use_video_file
        area.save()
        messages.success(request, f"Video source switched to {'Video File' if area.use_video_file else 'Video URL'}")
    return redirect('video_feed', area_id=area.id)

def generate_frames():
    """Generate video frames for streaming with YOLO-based plate detection."""
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("Error: Unable to access the camera.")

        plate_tracker = {}  # Store plate detections for temporal smoothing
        frame_count = 0
        process_interval = 5

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame.")
                break

            frame_count += 1
            if frame_count % process_interval == 0 and yolo_model:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = yolo_model.predict(rgb_frame, device='cpu')

                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        
                        # Draw rectangle
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        
                        # Extract plate region for OCR
                        plate_roi = frame[y1:y2, x1:x2]
                        preprocessed_plate = preprocess_plate_image(plate_roi)
                        
                        # Use EasyOCR for OCR
                        ocr_results = easyocr_reader.readtext(preprocessed_plate, detail=1)
                        for (bbox, detected_plate, ocr_confidence) in ocr_results:
                            detected_plate = ''.join(char for char in detected_plate if char.isalnum())
                            if not detected_plate or len(detected_plate) < 4:
                                continue

                            # Temporal smoothing
                            if detected_plate in plate_tracker:
                                plate_tracker[detected_plate]['count'] += 1
                                plate_tracker[detected_plate]['confidence'] = max(
                                    plate_tracker[detected_plate]['confidence'], ocr_confidence * confidence
                                )
                            else:
                                plate_tracker[detected_plate] = {'count': 1, 'confidence': ocr_confidence * confidence}

                            # Display if detected multiple times
                            if plate_tracker[detected_plate]['count'] >= 3:
                                display_text = f"{detected_plate} ({plate_tracker[detected_plate]['confidence'] * 100:.2f}%)"
                                cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except GeneratorExit:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"Error in video stream: {e}")
    finally:
        cap.release()

def video_feed1(request):
    """Stream video feed with YOLO-based plate detection."""
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_detected_plates(request, area_id):
    """Retrieve detected plates for a specific area."""
    plates = DetectedPlate.objects.filter(area_id=area_id).select_related('vehicle')
    plate_data = [
        {
            'plate': plate.plate,
            'classification': plate.classification,
            'confidence': plate.confidence,
            'vehicle': {
                'owner_name': plate.vehicle.owner_name if plate.vehicle else None,
                'make': plate.vehicle.make if plate.vehicle else None,
                'model': plate.vehicle.model if plate.vehicle else None,
            } if plate.vehicle else None
        }
        for plate in plates
    ]
    return JsonResponse({'plates': plate_data})