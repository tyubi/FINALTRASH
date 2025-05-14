from ultralytics import YOLO

# Load a model
model = YOLO("240_ecomp_yolov8n.pt")  # Load an official model or custom model

# Export the model
model.export(format="edgetpu")