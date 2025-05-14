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
# Database Configuration
# -------------------------------------
DB_CONFIG = {
    'host': 'trasshtech.mysql.database.azure.com',
    'user': 'trashtechdba',
    'password': '@Tr4shT3ch!',
    'database': 'trashtechdb',
}

def insert_detection(category, timestamp):
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

def insert_sensor_reading(sensor_id, reading_value, timestamp):
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        if connection.is_connected():
            cursor = connection.cursor()
            sql = """
            INSERT INTO sensor (sensor_id, reading_value, timestamp)
            VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (sensor_id, reading_value, timestamp))
            connection.commit()
            print(f"Inserted sensor reading: Sensor {sensor_id} = {reading_value} at {timestamp}")
    except Error as e:
        print(f"Error inserting sensor readings into DB: {e}")
        return False

# -------------------------------------
# Arduino Setup
# -------------------------------------
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)  # Wait for Arduino reset

component_queue = queue.Queue()
cooldown_tracker = defaultdict(lambda: 0)
cooldown_period = 5.0
active_objects = defaultdict(list)
recently_detected = defaultdict(list)

# -------------------------------------
# Arduino Communication (Sequential)
# -------------------------------------
arduino = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
time.sleep(2)

component_queue = queue.Queue()
cooldown_tracker = defaultdict(lambda: 0)
cooldown_period = 5.0
active_objects = defaultdict(list)
recently_detected = defaultdict(list)

# Arduino communication thread (sensor readings handled one-by-one)
def arduino_communication():
    while True:
        try:
            component = component_queue.get(timeout=1)
            if component:
                print(f"Sending to Arduino: {component.strip()}")
                arduino.write(component.encode('ascii', errors='ignore'))

                response_buffer = []
                response_timeout = time.time() + 4
                while time.time() < response_timeout:
                    if arduino.in_waiting > 0:
                        response = arduino.readline().decode('ascii', errors='ignore').strip()
                        if response:
                            print(f"Arduino response: {response}")
                            response_buffer.append(response)

                    if "DONE" in response_buffer:
                        break

                # Process each sensor type one-by-one
                for response in response_buffer:
                    if "CO2 Level" in response and "NH3 Level" in response:
                        co2_status = response.split('|')[0].split(':')[1].strip()
                        nh3_status = response.split('|')[1].split(':')[1].strip()

                        if co2_status == "TOXIC" or nh3_status == "TOXIC":
                            overall_status = "TOXIC"
                        elif co2_status == "ABOVE NORMAL" or nh3_status == "ABOVE NORMAL":
                            overall_status = "ABOVE NORMAL"
                        else:
                            overall_status = "NORMAL"

                        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        insert_sensor_reading(3, overall_status, timestamp_str)

                for response in response_buffer:
                    if "Container 1" in response:
                        container1_text = response.split(':')[1].strip()
                        container1_fill = int(container1_text.split('%')[0])
                        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        insert_sensor_reading(1, container1_fill, timestamp_str)
                        
                for response in response_buffer:
                    if "Container 2" in response:
                        container2_text = response.split(':')[1].strip()
                        container2_fill = int(container2_text.split('%')[0])
                        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        insert_sensor_reading(2, container2_fill, timestamp_str)
                        
        except queue.Empty:
            time.sleep(0.5)
        except Exception as e:
            print(f"Arduino communication error: {e}")
            time.sleep(0.5)

arduino_thread = threading.Thread(target=arduino_communication, daemon=True)
arduino_thread.start()

# -------------------------------------
# YOLOv8 Detection + Counting + Arduino
# -------------------------------------
original_classes = ['Biodegradable', 'Non-biodegradable', 'Recyclable']
count_classes = original_classes.copy()
model = YOLO('/home/thesis/Downloads/MGA MODELS/MODELS/AUGMENTED/v8_75ep_edgetpu.tflite', task='detect')

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam not detected.")
    exit()

frame_w, frame_h = 640, 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera initialized: {actual_w}x{actual_h}")

rect_w, rect_h = 400, 350
rect_x1 = (actual_w - rect_w) // 2
rect_y1 = (actual_h - rect_h) // 2
rect_x2 = rect_x1 + rect_w
rect_y2 = rect_y1 + rect_h

print("Detection started. Press 'q' to quit.")

object_presence_timeout = 1.0
last_detected_object = None
frame_clear_time = time.time()
counts = {cls: 0 for cls in count_classes}

def get_centroid(xyxy):
    x1, y1, x2, y2 = xyxy
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

while True:
    tic = time.time()
    ret, frame = cap.read()
    if not ret:
        break

    result = model(frame, imgsz=320, conf=0.7, verbose=False)
    detections = result[0].boxes
    annotated_frame = frame.copy()

    current_time = time.time()
    frame_is_clear = True

    if detections and detections.xyxy is not None and len(detections.xyxy) > 0:
        for i, box in enumerate(detections.xyxy):
            cls_id = int(detections.cls[i].item())
            class_name = original_classes[cls_id]
            if class_name not in counts:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = get_centroid((x1, y1, x2, y2))

            if rect_x1 <= cx <= rect_x2 and rect_y1 <= cy <= rect_y2:
                frame_is_clear = False
                if last_detected_object is None and current_time - frame_clear_time > object_presence_timeout:
                    counts[class_name] += 1
                    last_detected_object = (class_name, cx, cy)
                    print(f"Detected {class_name} at {datetime.now().strftime('%H:%M:%S')} - Count: {counts[class_name]}")

                    timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    insert_detection(class_name, timestamp_str)

                    if class_name == 'Biodegradable':
                        component_queue.put('BIO\n')
                    elif class_name == 'Non-biodegradable':
                        component_queue.put('NONBIO\n')
                    elif class_name == 'Recyclable':
                        component_queue.put('RECY\n')

                    time.sleep(3)

                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 255, 255), -1)

    if frame_is_clear:
        if last_detected_object is not None:
            frame_clear_time = current_time
            last_detected_object = None

    cv2.rectangle(annotated_frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (255, 0, 0), 2)
    y_offset = 30
    for i, (cls, count) in enumerate(counts.items()):
        cv2.putText(annotated_frame, f"{cls}: {count}", (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    fps = 1.0 / (time.time() - tic)
    cv2.putText(annotated_frame, f'FPS: {fps:.2f}', (annotated_frame.shape[1] - 150, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Detection & Sensor Interface', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
