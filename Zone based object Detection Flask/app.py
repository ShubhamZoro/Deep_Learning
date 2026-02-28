from flask import Flask, request, jsonify, render_template, send_file, Response
import cv2
import numpy as np
import os
import threading
import uuid
from werkzeug.utils import secure_filename
import json
from datetime import datetime
import time

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB

# Create directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Initialize YOLO model
model = None
yolo_loaded = False

# YOLO class mappings
class_ids = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3
}

def init_yolo_model():
    """Initialize YOLO model with yolo11m.pt"""
    global model, yolo_loaded
    try:
        print("🔄 Loading YOLO11m model...")
        
        from ultralytics import YOLO
        model = YOLO('yolo11m.pt')
        
        # Test the model
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        test_results = model(dummy_frame, verbose=False)
        
        yolo_loaded = True
        print("✅ YOLO11m model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ YOLO11m initialization failed: {e}")
        yolo_loaded = False
        return False

# Global variables
processing_status = {}
loaded_files = {}  # Store loaded file info
current_camera = None
camera_thread = None
camera_active = False
camera_frame = None
camera_zones = {'left_zone_percent': 50, 'right_zone_percent': 50}  # Changed to percentages

def process_frame(frame, left_zone_percent, right_zone_percent):
    """Process frame with zones using percentages"""
    if model is None or not yolo_loaded:
        return frame
    
    # Run detection
    results = model(frame, verbose=False)
    detections = results[0]

    height, width = frame.shape[:2]
    processed_frame = frame.copy()
    
    # Convert percentages to pixel coordinates
    left_zone_end = int(width * left_zone_percent / 100)
    right_zone_start = int(width * (100 - right_zone_percent) / 100)

    if detections.boxes is not None:
        boxes = detections.boxes.xyxy.cpu().numpy()
        classes = detections.boxes.cls.cpu().numpy()
        confidences = detections.boxes.conf.cpu().numpy()

        for box, cls, conf in zip(boxes, classes, confidences):
            if conf > 0.3 and int(cls) in class_ids.values():  # Fixed confidence threshold
                x1, y1, x2, y2 = map(int, box)
                class_name = model.names[int(cls)]

                # Calculate bounding box center
                box_center_x = (x1 + x2) // 2

                # Person detection (left zone only)
                if int(cls) == class_ids["person"] and box_center_x <= left_zone_end:
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(processed_frame, f"Person: {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Vehicle detection (right zone only)
                elif (int(cls) in [class_ids["bicycle"], class_ids["car"], class_ids["motorcycle"]]
                      and box_center_x >= right_zone_start):
                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(processed_frame, f"{class_name}: {conf:.2f}", (x1, y1-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Add zone dividers and labels
    if left_zone_end < right_zone_start:
        # Left zone divider
        cv2.line(processed_frame, (left_zone_end, 0), (left_zone_end, height), (255, 255, 255), 2)
        # Right zone divider
        cv2.line(processed_frame, (right_zone_start, 0), (right_zone_start, height), (255, 255, 255), 2)

        # Add zone labels with percentages
        cv2.putText(processed_frame, f"PERSON {left_zone_percent}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, "NEUTRAL", (left_zone_end + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(processed_frame, f"VEHICLE {right_zone_percent}%", (right_zone_start + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        # Adjacent zones
        cv2.line(processed_frame, (left_zone_end, 0), (left_zone_end, height), (255, 255, 255), 2)
        cv2.putText(processed_frame, f"PERSON {left_zone_percent}%", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(processed_frame, f"VEHICLE {right_zone_percent}%", (left_zone_end + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Optional: Add semi-transparent overlays to show zones
    overlay = processed_frame.copy()

    if left_zone_end < right_zone_start:
        # Green tint for person zone
        cv2.rectangle(overlay, (0, 0), (left_zone_end, height), (0, 255, 0), -1)
        # Blue tint for vehicle zone
        cv2.rectangle(overlay, (right_zone_start, 0), (width, height), (255, 0, 0), -1)
    else:
        # Adjacent zones
        cv2.rectangle(overlay, (0, 0), (left_zone_end, height), (0, 255, 0), -1)
        cv2.rectangle(overlay, (left_zone_end, 0), (width, height), (255, 0, 0), -1)

    # Blend overlays with low opacity
    processed_frame = cv2.addWeighted(processed_frame, 0.9, overlay, 0.1, 0)

    return processed_frame

def generate_camera_frames():
    """Generate camera frames for web streaming"""
    global camera_frame, camera_active, camera_zones
    
    while camera_active:
        if camera_frame is not None:
            # Process frame with current zones (using percentages)
            processed_frame = process_frame(
                camera_frame, 
                camera_zones['left_zone_percent'], 
                camera_zones['right_zone_percent']
            )
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        
        time.sleep(0.03)  # ~30 FPS

def camera_capture_thread():
    """Capture frames from camera"""
    global camera_frame, camera_active, current_camera
    
    try:
        current_camera = cv2.VideoCapture(0)
        if not current_camera.isOpened():
            print("❌ Could not open camera")
            return
        
        print("📹 Camera capture started")
        
        while camera_active:
            ret, frame = current_camera.read()
            if ret:
                camera_frame = frame
            else:
                break
            
            time.sleep(0.01)  # Small delay
        
        current_camera.release()
        print("📹 Camera capture stopped")
        
    except Exception as e:
        print(f"❌ Camera capture error: {e}")

def process_video_thread(job_id, video_source, left_zone_percent, right_zone_percent, input_source_type):
    """Process video in background thread"""
    try:
        processing_status[job_id]['status'] = 'processing'
        processing_status[job_id]['progress'] = 0
        
        print(f"🎬 Processing: {input_source_type}")
        print(f"   Zones: Person({left_zone_percent}% from left) | Vehicle({right_zone_percent}% from right)")
        
        # Initialize video capture
        if input_source_type == 'camera':
            cap = cv2.VideoCapture(0)
        elif input_source_type == 'rtsp':
            cap = cv2.VideoCapture(video_source)
        else:  # file
            cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            raise Exception("Could not open video source")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if input_source_type == 'file' else 1000
        
        print(f"📺 Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        
        # Setup output
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], f'{job_id}_processed.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with working detection code (using percentages)
            processed_frame = process_frame(frame, left_zone_percent, right_zone_percent)
            out.write(processed_frame)
            
            frame_count += 1
            
            # Update progress
            if input_source_type == 'file':
                progress = min(int((frame_count / total_frames) * 100), 100)
                processing_status[job_id]['progress'] = progress
            
            # Limit frames for live sources
            if input_source_type in ['camera', 'rtsp'] and frame_count >= 300:
                break
        
        cap.release()
        out.release()
        
        processing_status[job_id]['status'] = 'completed'
        processing_status[job_id]['progress'] = 100
        processing_status[job_id]['output_path'] = output_path
        
        print(f"✅ Processing completed: {output_path}")
        
    except Exception as e:
        print(f"❌ Processing failed: {str(e)}")
        processing_status[job_id]['status'] = 'error'
        processing_status[job_id]['error'] = str(e)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/load_file', methods=['POST'])
def load_file():
    """Load video file for processing"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file uploaded'}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected'}), 400
        
        # Save file
        filename = secure_filename(video_file.filename)
        file_id = str(uuid.uuid4())
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{file_id}_{filename}')
        video_file.save(video_path)
        
        print(f"📁 File uploaded: {filename} -> {video_path}")
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return jsonify({'error': 'Invalid video file'}), 400
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Store file info for processing
        loaded_files[file_id] = {
            'filename': filename,
            'path': video_path,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration
        }
        
        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'width': width,
            'height': height,
            'fps': fps,
            'frame_count': frame_count,
            'duration': duration,
            'message': 'File loaded successfully. Click "Start Processing" to begin.'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_processing', methods=['POST'])
def start_processing():
    """Start video processing with loaded file"""
    try:
        # Get parameters (now using percentages)
        file_id = request.json.get('file_id')
        left_zone_percent = int(request.json.get('left_zone_percent', 50))
        right_zone_percent = int(request.json.get('right_zone_percent', 50))
        
        if not file_id or file_id not in loaded_files:
            return jsonify({'error': 'No file loaded or file not found'}), 400
        
        file_info = loaded_files[file_id]
        video_path = file_info['path']
        
        # Create processing job
        job_id = str(uuid.uuid4())
        
        processing_status[job_id] = {
            'status': 'started',
            'progress': 0,
            'created_at': datetime.now().isoformat(),
            'left_zone_percent': left_zone_percent,
            'right_zone_percent': right_zone_percent,
            'input_source_type': 'file',
            'filename': file_info['filename']
        }
        
        # Start processing thread
        thread = threading.Thread(
            target=process_video_thread,
            args=(job_id, video_path, left_zone_percent, right_zone_percent, 'file')
        )
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'message': 'Video processing started',
            'file_info': file_info
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/start_camera', methods=['POST'])
def start_camera():
    """Start camera with web streaming"""
    global camera_thread, camera_active, camera_zones
    
    try:
        # Get zone parameters (now using percentages)
        left_zone_percent = int(request.json.get('left_zone_percent', 50))
        right_zone_percent = int(request.json.get('right_zone_percent', 50))
        
        # Update camera zones
        camera_zones = {
            'left_zone_percent': left_zone_percent,
            'right_zone_percent': right_zone_percent
        }
        
        print(f"📹 Starting camera with zones: Person({left_zone_percent}% from left) | Vehicle({right_zone_percent}% from right)")
        
        # Stop existing camera
        if camera_active:
            stop_camera()
        
        # Start camera capture thread
        camera_active = True
        camera_thread = threading.Thread(target=camera_capture_thread)
        camera_thread.daemon = True
        camera_thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Camera started with real-time detection',
            'stream_url': '/video_feed',
            'left_zone_percent': left_zone_percent,
            'right_zone_percent': right_zone_percent
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera"""
    global camera_active, current_camera, camera_frame
    
    try:
        camera_active = False
        camera_frame = None
        
        if current_camera is not None:
            current_camera.release()
            current_camera = None
        
        return jsonify({
            'status': 'stopped',
            'message': 'Camera stopped'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route for camera"""
    return Response(generate_camera_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get processing status"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(processing_status[job_id])

@app.route('/api/download/<job_id>')
def download_video(job_id):
    """Download processed video"""
    if job_id not in processing_status:
        return jsonify({'error': 'Job not found'}), 404
    
    status = processing_status[job_id]
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    output_path = status.get('output_path')
    if not output_path or not os.path.exists(output_path):
        return jsonify({'error': 'Output file not found'}), 404
    
    return send_file(output_path, as_attachment=True, download_name=f'processed_{job_id}.mp4')

if __name__ == '__main__':
    print("🎯 YOLO Zone Detection API Server")
    print("=" * 50)
    
    # Initialize YOLO model
    if init_yolo_model():
        print("✅ YOLO11m model ready for detection")
    else:
        print("⚠️  YOLO model failed to load")
    
    print(f"\n🚀 Server starting on http://localhost:5000")
    print("📹 Camera streaming available at: /video_feed")
    print("=" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)