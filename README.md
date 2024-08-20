# Real-Time Vehicle and Pedestrian Detection System

## Overview
This project aims to develop a real-time vehicle and pedestrian detection system to enhance driving safety. By utilizing the YOLOv9 object detection model, the system accurately identifies vehicles and pedestrians, estimates their distance from the vehicle, and triggers alerts when necessary. The system is designed to operate in real-time, providing critical information to drivers to prevent potential collisions.

## Features
- **Real-Time Object Detection**: Utilizes the YOLOv9 model to detect vehicles and pedestrians in video frames.
- **Distance Estimation**: Implements depth estimation algorithms to calculate the distance between the vehicle and detected objects.
- **Alert System**: Triggers visual and auditory alerts when pedestrians are detected within a critical distance threshold.
- **Responsive Frontend**: User interface developed using HTML, CSS, and JavaScript, allowing users to upload videos, view live camera feeds, and monitor real-time detection statistics.
- **Backend Integration**: Built with Flask, the backend handles video processing, object detection, and alert generation.

## System Architecture
- **Camera Integration**: Captures real-time video footage from a vehicle-mounted camera.
- **Object Detection**: YOLOv9 processes each frame to identify vehicles and pedestrians, providing bounding boxes and class labels.
- **Distance Estimation**: Estimates the distance of detected objects based on bounding box sizes and known dimensions.
- **Alert Mechanism**: Configurable thresholds trigger alerts when objects are within a predefined proximity to the vehicle.

## UML Diagrams
The system’s structure and interactions are represented using UML diagrams, including:
- **Use Case Diagrams**
- **Class Diagrams**
- **Sequence Diagrams**
- **Activity Diagrams**

## Development
### Frontend
- **Video and Photo Upload**: Allows users to upload media files for analysis.
- **Live Camera Feed**: Displays real-time video from the camera.
- **Statistics Display**: Shows real-time detection statistics and alerts.
- **Technology Stack**: HTML, CSS, JavaScript.

### Backend
- **Flask Server**: Handles server-side logic, including video processing and object detection.
- **YOLOv9 Integration**: Detects vehicles and pedestrians in video frames.
- **Alert System**: Generates notifications when objects are within a critical distance.
- **Technology Stack**: Python, Flask, OpenCV, YOLOv9.

### Real-Time Processing
- **Frame Capture**: Captures frames from the camera feed using OpenCV.
- **Object Detection**: Analyzes each frame with YOLOv9.
- **Distance Estimation**: Uses bounding box dimensions to estimate object distance.

## Installation
1. **Clone the Repository**:
    ```bash
    git clone https://github.com/RohithgowdM/Vehicle-pedestrian-detection.git
    cd vehicle-pedestrian-detection
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the Application**:
    ```bash
    python app.py
    ```

## Usage
1. **Upload Video**: Upload a video file or use a live camera feed.
2. **Start Detection**: Click on the start button to begin real-time detection.
3. **Monitor Alerts**: The system will alert you if any object is detected within the critical distance.

## Testing
- **Unit Testing**: Individual components like the object detection and distance estimation functions are tested in isolation.
- **Integration Testing**: The interaction between the detection module, distance estimation, and alert system is validated.
- **Real-World Testing**: System performance is evaluated under various driving conditions, including different lighting, weather, and traffic situations.

## Future Enhancements
- **Multilingual Support**: Adding support for multiple languages in the user interface.
- **Advanced Distance Estimation**: Implementing more sophisticated depth estimation techniques.
- **Expanded Object Classes**: Extending detection capabilities to other objects like cyclists and road signs.

## Contributing
Contributions are welcome! Please follow the standard [contribution guidelines](CONTRIBUTING.md) and adhere to the code of conduct.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements
- **YOLOv9**: For the object detection model.
- **Flask**: For the web framework.
- **OpenCV**: For the image processing library.
