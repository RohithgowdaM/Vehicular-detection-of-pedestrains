from flask import Flask, render_template, request, Response, jsonify
import cv2
import threading
import os
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import numpy as np
import winsound

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load YOLOv9 model
model = YOLO('./yolov9c.pt')

# Classes
human_class = 0
vehicle_classes = [1, 2, 3, 5, 7]

camera_active = False

# Define constants for distance estimation
KNOWN_WIDTH = 0.5  # Average shoulder width in meters (adjust based on your context)
FOCAL_LENGTH = 700  # Camera focal length in pixels (adjust based on calibration)

# Optical Flow parameters
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Previous frame and points
prev_frame = None
prev_points = None

frame_count = 0

# Summary of detections
detections_summary = {
    "total_detections": 0,
    "vehicles_detected": 0,
    "pedestrians_detected": 0
}

def estimate_distance(box):
    """
    Estimate distance to the object based on the width of the bounding box.
    """
    box_width = box[2] - box[0]
    distance = (KNOWN_WIDTH * FOCAL_LENGTH) / box_width
    return distance

def is_near(distance, threshold=4.0):
    """
    Determine if the object is near based on the distance threshold.
    """
    return distance < threshold

def send_alert():
    winsound.Beep(1000, 500)  # Frequency and duration of the beep

def track_objects(frame, detections):
    global prev_frame, prev_points

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if prev_frame is None:
        prev_frame = gray_frame

    # Resize the gray_frame to match the size of prev_frame if they differ
    if prev_frame.shape != gray_frame.shape:
        gray_frame = cv2.resize(gray_frame, (prev_frame.shape[1], prev_frame.shape[0]))

    # Initialize tracking points if there are new detections
    if detections:
        new_points = []
        for result in detections.boxes:
            box = result.xyxy[0].cpu().numpy()
            cls = int(result.cls[0].cpu().numpy())
            if cls in [human_class] + vehicle_classes:
                new_points.append([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])
        if new_points:
            prev_points = np.float32(new_points).reshape(-1, 1, 2)

    # Calculate optical flow if there are valid points
    try:
        if prev_points is not None and prev_points.size > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(prev_frame, gray_frame, prev_points, None, **lk_params)
            for i, (new, old) in enumerate(zip(next_points, prev_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                speed = np.sqrt((a - c) ** 2 + (b - d) ** 2)
                direction = np.arctan2(b - d, a - c) * 180 / np.pi
                cv2.circle(frame, (int(a), int(b)), 5, (0, 255, 0), -1)
                cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                cv2.putText(frame, f'Speed: {speed:.2f} Direction: {direction:.2f}', (int(a), int(b)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            prev_points = next_points
    except cv2.error as e:
        print(f"Optical flow error: {e}")
        prev_points = None  # Reset the points on error

    prev_frame = gray_frame
    return frame

def detect_and_alert(frame):
    global detections_summary
    global frame_count

    if frame_count % 1 == 0:  # Process every 4th frame
        results = model.predict(frame)[0]
        alerts = []

        frame = track_objects(frame, results)

        for result in results.boxes:
            box = result.xyxy[0].cpu().numpy()
            cls = int(result.cls[0].cpu().numpy())
            conf = result.conf[0].cpu().numpy()

            if cls == human_class:
                distance = estimate_distance(box)
                if is_near(distance):
                    alerts.append(box)
                label = model.names[int(cls)]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f} Dist: {distance:.2f}m', (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                detections_summary["pedestrians_detected"] += 1
            elif cls in vehicle_classes:
                label = model.names[int(cls)]
                cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detections_summary["vehicles_detected"] += 1

        detections_summary["total_detections"] = detections_summary["pedestrians_detected"] + detections_summary["vehicles_detected"]

        if alerts:
            threading.Thread(target=send_alert).start()

    frame_count += 1
    return frame

def generate_frames():
    global camera_active
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    while camera_active:
        success, frame = cap.read()
        if not success:
            print("Error: Failed to read frame from camera.")
            break
        frame = detect_and_alert(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def detect_file(filepath):
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = detect_and_alert(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

def detect_photo(filepath):
    """
    Detect objects in a photo.
    """
    frame = cv2.imread(filepath)
    if frame is None:
        print("Error: Could not read image.")
        return None

    # Perform detection and return the annotated frame
    frame = detect_and_alert(frame)
    ret, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global camera_active
    if camera_active:
        return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return "", 204

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return ("Camera turned on" if camera_active else "Camera turned off"), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return filename

@app.route('/upload_video_feed', methods=['POST'])
def upload_video_feed():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return Response(detect_file(os.path.join(app.config['UPLOAD_FOLDER'], filename)), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_photo', methods=['POST'])
def upload_photo():
    """
    Endpoint to upload a photo and return detected objects on the image.
    """
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Detect objects in the photo
        detected_image = detect_photo(filepath)

        if detected_image is not None:
            return Response(detected_image, mimetype='image/jpeg')
        else:
            return 'Error processing image', 500

@app.route('/stats')
def stats():
    global detections_summary
    return jsonify(detections_summary)

if __name__ == '__main__':
    app.run(debug=True)
