from ultralytics import YOLO
import cv2
import time

# Load the YOLOv8 model
model = YOLO("YOLOv8s.pt")

# Open the webcam or video capture (0 = default webcam, you can change it to 1, 2, etc., for other cameras)
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

print("Press 'q' to quit the live stream.")

# Initialize variables for calculating FPS
prev_time = 0

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Start time for FPS calculation
    current_time = time.time()

    # Run YOLO inference on the frame
    results = model(frame)

    # Visualize the results on the frame
    annotated_frame = results[0].plot()  # Annotate the frame with bounding boxes, labels, etc.

    # Calculate FPS
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # Display FPS on the frame
    cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Live Detection", annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
