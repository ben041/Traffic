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
    yolo_model = YOLO('media/models/best.pt')
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    yolo_model = None

# Initialize EasyOCR
try:
    easyocr_reader = easyocr.Reader(['en'], gpu=False)
except Exception as e:
    logger.error(f"Error initializing EasyOCR: {e}")
    easyocr_reader = None

def preprocess_plate_image(plate_img):
    """Preprocess license plate image for OCR."""
    try:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        return gray
    except Exception as e:
        logger.error(f"Error preprocessing plate image: {e}")
        return plate_img

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
    return render(request, 'vehicle_list.html', context)

@login_required
def vehicle_detail(request, plate_number):
    """Display details of a specific vehicle."""
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    context = {
        'vehicle': vehicle,
        'today': timezone.now().date(),
    }
    return render(request, 'vehicle_detail.html', context)

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
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_rgb = cv2.resize(image_rgb, (320, 240))  # Resize for faster processing
                results = yolo_model.predict(image_rgb, device='cpu', conf=0.4)
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
                    ocr_results = easyocr_reader.readtext(plate_img, detail=1) if easyocr_reader else []
                    if not ocr_results:
                        tesseract_text = pytesseract.image_to_string(plate_img, config='--psm 8')
                        text = ''.join(e for e in tesseract_text if e.isalnum()).upper()
                        ocr_confidence = 0.5
                        ocr_results = [((0, 0, 0, 0), text, ocr_confidence)]
                    for (bbox, text, ocr_confidence) in ocr_results:
                        text = ''.join(e for e in text if e.isalnum()).upper()
                        if len(text) > 4:
                            results.append((text, ocr_confidence * confidence))
                return results
            except Exception as e:
                logger.error(f"Error in detect_plate: {str(e)}")
                return []

        try:
            if request.FILES.get('image'):
                image_file = request.FILES['image']
                image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
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
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame = cv2.resize(rgb_frame, (320, 240))
        results = yolo_model.predict(rgb_frame, device='cpu', conf=0.4)
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                plate_img = frame[y1:y2, x1:x2]
                preprocessed_plate = preprocess_plate_image(plate_img)
                ocr_results = easyocr_reader.readtext(preprocessed_plate, detail=1) if easyocr_reader else []
                if not ocr_results:
                    tesseract_text = pytesseract.image_to_string(preprocessed_plate, config='--psm 8')
                    plate_text = ''.join(char for char in tesseract_text if char.isalnum()).upper()
                    ocr_confidence = 0.5
                    ocr_results = [((0, 0, 0, 0), plate_text, ocr_confidence)]
                for (bbox, plate_text, ocr_confidence) in ocr_results:
                    plate_text = ''.join(char for char in plate_text if char.isalnum()).upper()
                    if len(plate_text) < 4:
                        continue
                    try:
                        vehicle = Vehicle.objects.get(plate_number=plate_text)
                        suspect_vehicle = SuspectVehicle.objects.filter(vehicle=vehicle, is_active=True).first()
                        classification = "Suspect" if suspect_vehicle else "Not Suspect"
                        detection = PlateDetection.objects.create(
                            vehicle=vehicle,
                            detected_plate=plate_text,
                            confidence=ocr_confidence * confidence,
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

def generate_frames(area_id=None):
    """Generate video frames for streaming with YOLO-based plate detection."""
    debug_dir = "debug_frames"
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    try:
        # Try different webcam indices
        for index in [0, 1, 2]:
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                logger.info(f"Webcam opened successfully on index {index}")
                break
        else:
            logger.error("Error: No webcam found")
            raise Exception("No webcam found")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        area = None
        if area_id:
            try:
                area = Area.objects.get(id=area_id)
            except Area.DoesNotExist:
                logger.warning(f"Area with id {area_id} not found")
                area, _ = Area.objects.get_or_create(
                    id=area_id,
                    defaults={'name': f'Area {area_id}', 'description': 'Default area'}
                )
        plate_tracker = {}
        frame_count = 0
        process_interval = 15
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame")
                break
            frame_count += 1
            if frame_count % process_interval == 0 and yolo_model and easyocr_reader:
                try:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    rgb_frame = cv2.resize(rgb_frame, (320, 240))
                    results = yolo_model.predict(rgb_frame, device='cpu', conf=0.4)
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            confidence = box.conf[0]
                            # Scale coordinates back to original frame
                            x1, y1, x2, y2 = [int(coord * 640/320) for coord in [x1, y1, x2, y2]]
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            plate_roi = frame[y1:y2, x1:x2]
                            if plate_roi.size == 0:
                                logger.warning("Empty plate ROI detected")
                                continue
                            preprocessed_plate = preprocess_plate_image(plate_roi)
                            # Save debug image
                            debug_path = os.path.join(debug_dir, f"plate_{frame_count}.jpg")
                            cv2.imwrite(debug_path, preprocessed_plate)
                            logger.info(f"Saved debug plate image: {debug_path}")
                            ocr_results = easyocr_reader.readtext(preprocessed_plate, detail=1)
                            if not ocr_results:
                                tesseract_text = pytesseract.image_to_string(preprocessed_plate, config='--psm 8')
                                detected_plate = ''.join(char for char in tesseract_text if char.isalnum()).upper()
                                ocr_confidence = 0.5
                                ocr_results = [((0, 0, 0, 0), detected_plate, ocr_confidence)]
                            for (bbox, detected_plate, ocr_confidence) in ocr_results:
                                detected_plate = ''.join(char for char in detected_plate if char.isalnum()).upper()
                                if len(detected_plate) < 4:
                                    logger.info(f"Skipped short plate: {detected_plate}")
                                    continue
                                logger.info(f"Detected plate: {detected_plate}, confidence: {ocr_confidence * confidence}")
                                if detected_plate not in plate_tracker:
                                    plate_tracker[detected_plate] = {'count': 0, 'confidence': 0}
                                plate_tracker[detected_plate]['count'] += 1
                                plate_tracker[detected_plate]['confidence'] = max(
                                    plate_tracker[detected_plate]['confidence'], ocr_confidence * confidence
                                )
                                if plate_tracker[detected_plate]['confidence'] > 0.3:
                                    try:
                                        vehicle = Vehicle.objects.filter(plate_number=detected_plate).first()
                                        classification = "Unknown"
                                        if vehicle:
                                            suspect_vehicle = SuspectVehicle.objects.filter(
                                                vehicle=vehicle, is_active=True
                                            ).first()
                                            classification = "Suspect" if suspect_vehicle else "Not Suspect"
                                        DetectedPlate.objects.get_or_create(
                                            plate=detected_plate,
                                            defaults={
                                                'classification': classification,
                                                'vehicle': vehicle,
                                                'area_id': area_id if area_id else 0,
                                                'confidence': plate_tracker[detected_plate]['confidence']
                                            }
                                        )
                                        logger.info(f"Saved DetectedPlate: {detected_plate}, area_id: {area_id}")
                                        PlateDetection.objects.create(
                                            vehicle=vehicle,
                                            detected_plate=detected_plate,
                                            confidence=plate_tracker[detected_plate]['confidence'],
                                            area=area,
                                            timestamp=timezone.now()
                                        )
                                        logger.info(f"Saved PlateDetection: {detected_plate}")
                                        display_text = f"{detected_plate} ({plate_tracker[detected_plate]['confidence'] * 100:.2f}%)"
                                        cv2.putText(frame, display_text, (x1, y1 - 10),
                                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                    except Exception as e:
                                        logger.error(f"Error saving plate {detected_plate}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Error processing frame {frame_count}: {e}")
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    except Exception as e:
        logger.error(f"Error in video stream: {e}")
    finally:
        cap.release()

@login_required
def video_feed1(request, area_id=None):
    """Stream video feed with YOLO-based plate detection."""
    try:
        return StreamingHttpResponse(generate_frames(area_id),
                                    content_type='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        logger.error(f"Error starting video feed: {e}")
        return JsonResponse({'error': 'Failed to start video feed'}, status=500)

@login_required
def get_detected_plates(request, area_id):
    """Retrieve detected plates for a specific area."""
    try:
        plates = DetectedPlate.objects.filter(area_id=area_id).select_related('vehicle')
        logger.info(f"Queried DetectedPlate for area_id {area_id}, found {plates.count()} plates")
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
    except Exception as e:
        logger.error(f"Error retrieving detected plates: {e}")
        return JsonResponse({'error': 'Failed to retrieve detected plates'}, status=500)