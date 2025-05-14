import time
import cv2
from ultralytics import YOLO

# Custom class labels from your dataset
class_names = ['BJT', 'LED', 'burnt', 'capacitor', 'cracked', 'defective', 'faded', 'missing-leg', 'resistor', 'rust']

# Load the Edge TPU-compiled YOLOv8 TFLite model
try:
    print("Loading model...")
    model = YOLO('ecomp_yolov8_int8.tflite', task='detect')
    print("✅ Model loaded. If no TPU delegate error appeared, it's likely active.")
except Exception as e:
    print("❌ Failed to load model or TPU delegate:", e)
    exit(1)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Failed to open webcam.")
    exit(1)

print('✅ Live inference started. Press "q" to exit.')

while True:
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Failed to read frame from camera.")
        break

    try:
        # Run inference
        output = model(frame, imgsz=320, verbose=False)
    except Exception as e:
        print("❌ Inference error:", e)
        break

    # Draw detections
    for det in output[0].boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = det
        if score < 0.6:
            continue

        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        label = class_names[int(class_id)] if int(class_id) < len(class_names) else f"ID {int(class_id)}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} ({score:.2f})', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display FPS
    fps = 1.0 / (time.time() - tic)
    cv2.putText(frame, f'FPS: {fps:.2f}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Show the frame
    cv2.imshow('Edge TPU Live Inference', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
