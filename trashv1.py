import os
import time
import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime
import serial
import threading
import queue
from collections import defaultdict
import mysql.connector
from mysql.connector import Error

# -------------------------------------
# Database Configuration & Insert Function
# -------------------------------------
DB_CONFIG = {
    'host': 'trasshtech.mysql.database.azure.com',
    'user': 'trashtechdba@trasshtech',  # Add server name
    'password': '@Tr4shT3ch!',
    'database': 'trashtechdb',
    'port': 3306,
    'ssl_disabled': True  # Disable SSL for testing
}

def insert_detection(category, timestamp):
    """Insert detected category and timestamp into the trash table."""
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            sql = "INSERT INTO trash (category, timestamp) VALUES (%s, %s)"
            cursor.execute(sql, (category, timestamp))
            connection.commit()
            cursor.close()
            connection.close()
            print(f"Inserted into DB: {category} at {timestamp}")
            return True
    except Error as e:
        print(f"Error inserting into database: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

# -------------------------------------
# Arduino Setup
# -------------------------------------
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Change COM port if needed
time.sleep(2)  # Allow Arduino time to reset

# Queue and cooldown management
component_queue = queue.Queue()
cooldown_tracker = defaultdict(lambda: 0)
cooldown_period = 5.0  # Increase cooldown period

# Active object tracker
active_objects = defaultdict(list)  # class -> list of (cx, cy, timestamp)

# Add a new variable to track recently detected objects
recently_detected = defaultdict(list)  # class -> list of (cx, cy, timestamp)

# Arduino communication thread
def arduino_communication():
    while True:
        try:
            component = component_queue.get(timeout=1)
            if component:
                print(f"Sending to Arduino: {component.strip()}")
                arduino.write(component.encode('ascii', errors='ignore'))

                # Wait for Arduino reply
                response_timeout = time.time() + 2
                while time.time() < response_timeout:
                    if arduino.in_waiting > 0:
                        response = arduino.readline().decode('ascii', errors='ignore').strip()
                        print(f"Arduino response: {response}")
                        if response == "DONE":
                            break
                else:
                    print("Arduino response timeout.")
        except queue.Empty:
            time.sleep(0.5)
        except Exception as e:
            print(f"Arduino communication error: {e}")
            time.sleep(0.5)

# Start Arduino communication thread
arduino_thread = threading.Thread(target=arduino_communication, daemon=True)
arduino_thread.start()

# -------------------------------------
# YOLOv8 Detection + Counting + Arduino
# -------------------------------------
original_classes = ['Biodegradable', 'Non-biodegradable', 'Recyclable']
count_classes = ['Biodegradable', 'Non-biodegradable', 'Recyclable']

model = YOLO('/home/thesis/Downloads/200EPmodel_- 12 may 2025 19_01_edgetpu.tflite', task='detect')

# Get model info to check input size
model_input_size = 256  # Set to match your quantized model's input size

# Set camera resolution closer to model input size
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Failed to open webcam.")
    exit()

# Set fixed frame dimensions to smaller size
frame_w = 640  # Reduced from 640
frame_h = 480  # Reduced from 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

# Get actual frame dimensions
actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Frame dimensions set to: Width = {actual_w}, Height = {actual_h}")

# Define central rectangle area (centered box)
rect_w = 400  # Width of trigger box
rect_h = 350  # Height of trigger box
rect_x1 = (actual_w - rect_w) // 2
rect_y1 = (actual_h - rect_h) // 2
rect_x2 = rect_x1 + rect_w
rect_y2 = rect_y1 + rect_h

print("Live inference with counting started. Press 'q' to quit.")

# Modify the tracking parameters
min_distance = 50  # Distance threshold for object tracking
object_presence_timeout = 1.0  # Time to wait before considering object removed
last_detected_object = None  # Track the last detected object
frame_clear_time = time.time()  # Track when frame was last clear

# Initialize tracking variables
counts = {cls: 0 for cls in count_classes}
seen_centroids = {cls: [] for cls in count_classes}
last_detection_time = time.time()  # Initialize last_detection_time

def get_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while True:
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    # Fix the model inference line
    result = model(frame, imgsz=256, conf=0.6, verbose=False)  # Remove the error message that got mixed in

    detections = result[0].boxes
    annotated_frame = frame.copy()  # Start with clean frame

    current_time = time.time()
    frame_is_clear = True  # Assume frame is clear initially

    if detections is not None and detections.xyxy is not None and len(detections.xyxy) > 0:
        for i, box in enumerate(detections.xyxy):
            cls_id = int(detections.cls[i].item())
            class_name = original_classes[cls_id]

            if class_name not in counts:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = get_centroid((x1, y1, x2, y2))

            # Only process objects in the detection box
            if rect_x1 <= cx <= rect_x2 and rect_y1 <= cy <= rect_y2:
                frame_is_clear = False  # Object detected in frame
                
                if last_detected_object is None and current_time - frame_clear_time > object_presence_timeout:
                    # New detection after frame was clear
                    counts[class_name] += 1
                    last_detected_object = (class_name, cx, cy)
                    print(f"Detected {class_name} at {datetime.now().strftime('%H:%M:%S')} - Count: {counts[class_name]}")

                    # Insert detection into database
                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    insert_detection(class_name, timestamp_str)

                    # Send command to Arduino
                    if class_name == 'Biodegradable':
                        component_queue.put('BIO\n')
                    elif class_name == 'Non-biodegradable':
                        component_queue.put('NONBIO\n')
                    elif class_name == 'Recyclable':
                        component_queue.put('RECY\n')
                    
                    time.sleep(3)  # Brief delay after detection

                # Draw detection box and label
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)

    if frame_is_clear:
        if last_detected_object is not None:
            frame_clear_time = current_time
            last_detected_object = None
    
    # Draw central trigger rectangle
    cv2.rectangle(annotated_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)

    # Display counts
    y_offset = 30
    for i, (cls, count) in enumerate(counts.items()):
        cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display FPS
    fps = 1.0 / (time.time() - tic)
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (annotated_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Edge TPU Detection + Counting', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
