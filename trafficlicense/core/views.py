from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, StreamingHttpResponse
from django.contrib import messages
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
import cv2
import numpy as np
import pytesseract
import requests
import tempfile
import os
import logging
from .models import Area, Vehicle, PlateDetection, SuspectVehicle, DetectedPlate
from .forms import AreaForm, VehicleForm, SuspectVehicleForm, VideoUploadForm

# Configure logging
logger = logging.getLogger(__name__)

# Set Tesseract OCR path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Load Haar cascade for plate detection
PLATE_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml')


# Utility Functions
def download_video_from_url(video_url):
    """
    Download video from a URL and save to a temporary file.
    Returns the path to the temporary file or None if download fails.
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
        logger.error(f"Failed to download video: {e}")
        return None

def preprocess_plate_image(plate_img):
    """
    Preprocess the plate image to enhance OCR accuracy.
    Returns the preprocessed grayscale image.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    gray = cv2.medianBlur(gray, 3)
    return gray

def detect_and_classify_plates(video_path, area):
    """
    Detect and classify number plates from a video.
    Returns a list of detected plates with their details.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []

    detected_plates = []
    custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    frame_count = 0
    detection_interval = 10  # Process every 10th frame to optimize performance

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % detection_interval != 0:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = PLATE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(75, 25))

        for (x, y, w, h) in plates:
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
                logger.error(f"OCR processing error: {e}")

    cap.release()
    return detected_plates


# View Functions
def home(request):
    """
    Render the dashboard with statistics and vehicle list.
    """
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
    """
    Display a list of all camera installation areas.
    """
    areas = Area.objects.all()
    return render(request, 'area_list.html', {'areas': areas})

def suspected_vehicles(request):
    """
    Display a list of all suspected vehicles.
    """
    suspected_list = SuspectVehicle.objects.select_related('vehicle').all()
    return render(request, 'suspected_vehicles.html', {'suspected_vehicles': suspected_list})

def all_vehicles(request):
    """
    Display a list of all registered vehicles.
    """
    vehicles = Vehicle.objects.all()
    return render(request, 'all_vehicles.html', {'vehicles': vehicles})

def vehicle_details(request, plate_number):
    """
    Display details of a specific vehicle, including suspect status.
    """
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    suspect_details = SuspectVehicle.objects.filter(vehicle=vehicle).first()
    return render(request, 'vehicle_details.html', {
        'vehicle': vehicle,
        'suspect_details': suspect_details or SuspectVehicle(vehicle=vehicle)
    })

@login_required
def video_feed(request, area_id):
    """
    Render the video feed page for a specific area.
    """
    area = get_object_or_404(Area, id=area_id)
    context = {'area': area}
    return render(request, 'video_feed.html', context)

@login_required
def start_plate_detection(request, area_id):
    """
    Start number plate detection for a specific area and return detected plates as JSON.
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
    Toggle between video file and video URL for a specific area.
    """
    area = get_object_or_404(Area, id=area_id)
    if request.method == 'POST':
        area.use_video_file = not area.use_video_file
        area.save()
        messages.success(request, f"Video source switched to {'Video File' if area.use_video_file else 'Video URL'}")
    return redirect('video_feed', area_id=area.id)

@login_required
def add_edit_area(request, area_id=None):
    """
    Add or edit an area. If area_id is provided, edit the existing area.
    """
    area = get_object_or_404(Area, id=area_id) if area_id else None
    if request.method == 'POST':
        form = AreaForm(request.POST, request.FILES, instance=area)
        if form.is_valid():
            form.save()
            messages.success(request, 'Area saved successfully!')
            return redirect('area_list')
    else:
        form = AreaForm(instance=area)
    return render(request, 'add_edit_area.html', {'form': form, 'area': area})

@login_required
def add_edit_vehicle(request, plate_number=None):
    """
    Add or edit a vehicle. If plate_number is provided, edit the existing vehicle.
    """
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number) if plate_number else None
    if request.method == 'POST':
        form = VehicleForm(request.POST, instance=vehicle)
        if form.is_valid():
            form.save()
            messages.success(request, f"Vehicle {'updated' if vehicle else 'added'} successfully!")
            return redirect('all_vehicles')
    else:
        form = VehicleForm(instance=vehicle)
    return render(request, 'add_edit_vehicle.html', {'form': form, 'vehicle': vehicle})

@login_required
def add_edit_suspect_vehicle(request, plate_number):
    """
    Add or edit a suspect vehicle for a given vehicle plate number.
    """
    suspect_vehicle = SuspectVehicle.objects.filter(vehicle__plate_number=plate_number).first()
    if not suspect_vehicle:
        vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
        suspect_vehicle = SuspectVehicle(vehicle=vehicle)

    if request.method == 'POST':
        form = SuspectVehicleForm(request.POST, instance=suspect_vehicle)
        if form.is_valid():
            form.save()
            messages.success(request, 'Suspect Vehicle saved successfully!')
            return redirect('suspected_vehicles')
    else:
        form = SuspectVehicleForm(instance=suspect_vehicle)
    return render(request, 'add_edit_suspect_vehicle.html', {'form': form, 'suspect_vehicle': suspect_vehicle})

@login_required
def delete_area(request, area_id):
    """
    Delete an area after confirmation.
    """
    area = get_object_or_404(Area, id=area_id)
    if request.method == "POST":
        area.delete()
        messages.success(request, "Area deleted successfully!")
        return redirect('area_list')
    return render(request, 'confirm_delete.html', {'object': area, 'type': 'Area'})

@login_required
def delete_vehicle(request, plate_number):
    """
    Delete a vehicle after confirmation.
    """
    vehicle = get_object_or_404(Vehicle, plate_number=plate_number)
    if request.method == "POST":
        vehicle.delete()
        messages.success(request, "Vehicle deleted successfully!")
        return redirect('all_vehicles')
    return render(request, 'confirm_delete.html', {'object': vehicle, 'type': 'Vehicle'})

@login_required
def delete_suspect_vehicle(request, plate_number):
    """
    Delete a suspect vehicle after confirmation.
    """
    suspect_vehicle = get_object_or_404(SuspectVehicle, vehicle__plate_number=plate_number)
    if request.method == "POST":
        suspect_vehicle.delete()
        messages.success(request, "Suspect Vehicle deleted successfully!")
        return redirect('suspected_vehicles')
    return render(request, 'confirm_delete.html', {'object': suspect_vehicle, 'type': 'Suspect Vehicle'})

def signup_view(request):
    """
    Handle user signup and create a new user account.
    """
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
    """
    Handle user login and authenticate credentials.
    """
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user is not None:
            login(request, user)
            messages.success(request, "Logged in successfully.")
            return redirect('home')
        else:
            messages.error(request, "Invalid username or password.")
    return render(request, 'signin.html')

def logout_view(request):
    """
    Handle user logout and redirect to signin page.
    """
    logout(request)
    messages.success(request, "Logged out successfully.")
    return redirect('signin')

def generate_frames():
    """
    Generate video frames from the camera for live streaming with plate detection.
    Yields JPEG-encoded frames with detected plates.
    """
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            raise Exception("Unable to access the camera.")

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.error("Failed to capture frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            plates = PLATE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            for (x, y, w, h) in plates:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                plate_roi = frame[y:y+h, x:x+w]
                plate_roi = preprocess_plate_image(plate_roi)

                ocr_data = pytesseract.image_to_data(plate_roi, config='--psm 7', output_type=pytesseract.Output.DICT)
                detected_text = ocr_data['text']
                confidences = ocr_data['conf']

                max_confidence = 0
                detected_plate = ''
                for text, conf in zip(detected_text, confidences):
                    if text.strip() and int(float(conf)) > max_confidence:
                        max_confidence = int(float(conf))
                        detected_plate = text.strip()

                if detected_plate:
                    logger.info(f"Detected Plate: {detected_plate}, Confidence: {max_confidence}%")
                    display_text = f"{detected_plate} ({max_confidence}%)"
                    cv2.putText(frame, display_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

            _, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        logger.error(f"Error in video stream: {e}")
    finally:
        cap.release()

def video_feed1(request):
    """
    Stream live video feed with number plate detection.
    """
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def get_detected_plates(request, area_id):
    """
    Retrieve detected plates for a specific area and return as JSON.
    """
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

def gis_view(request):
    """
    Render the GIS visualization page.
    """
    return render(request, 'gis.html')