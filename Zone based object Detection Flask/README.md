# Zone-Based Object Detection Flask App

A Flask-based real-time object detection system using YOLO11m with zone-based filtering for person and vehicle detection.

## Features

- **Zone-Based Detection**: Split the frame into two zones:
  - Left zone: Detects persons only
  - Right zone: Detects vehicles (bicycle, car, motorcycle)
- **Real-time Camera Streaming**: Live webcam detection with adjustable zone percentages
- **Video Processing**: Upload and process video files
- **Adjustable Zones**: Configure zone percentages via API

## Requirements

- Python 3.8+
- Webcam or video file for detection

## Installation

1. Create and activate a virtual environment:
   ```bash
   python -m venv venva
   venva\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download YOLO11m model:
   - Place `yolo11m.pt` in the project root directory
   - Download from: https://docs.ultralytics.com/models/yolo11/

## Usage

1. Start the Flask server:
   ```bash
   python app.py
   ```

2. Open browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Use the web interface to:
   - Start live camera detection
   - Upload and process video files
   - Adjust zone percentages

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main interface |
| `/api/load_file` | POST | Upload video file |
| `/api/start_processing` | POST | Start video processing |
| `/api/start_camera` | POST | Start camera streaming |
| `/api/stop_camera` | POST | Stop camera streaming |
| `/video_feed` | GET | Camera stream endpoint |
| `/api/status/<job_id>` | GET | Get processing status |
| `/api/download/<job_id>` | GET | Download processed video |

## Zone Configuration

- `left_zone_percent`: Percentage of frame for person detection (from left)
- `right_zone_percent`: Percentage of frame for vehicle detection (from right)

## Project Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── templates/
│   └── index.html     # Web interface
├── uploads/           # Uploaded video files
├── outputs/           # Processed video files
└── yolo11m.pt         # YOLO model file
```
