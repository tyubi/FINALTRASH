from ultralytics import YOLO

# Load a model
model = YOLO("240_yolov8n_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model

# Run Prediction
model.predict("crowd.jpg", save=True, imgsz=240, conf=0.5)  # Inference defaults to the first TPU