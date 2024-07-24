from ultralytics import YOLO

# Load the YOLOv9 model
model = YOLO('./yolov9c.pt')

# Run predictions
results = model.predict(f'R:\RGM\photos\BR HILLS\IMG-20220716-WA0036.jpg')

# Display results
results[0].show()