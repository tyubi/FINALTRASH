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
import logging
from contextlib import contextmanager

# -------------------------------------
# Logging Configuration
# -------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('inference.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------------
# Database Configuration
# -------------------------------------
DB_CONFIG = {
    'host': 'trasshtech.mysql.database.azure.com',
    'user': 'trashtechdba',
    'password': '@Tr4shT3ch!',
    'database': 'trashtechdb',
    'pool_name': 'mypool',
    'pool_size': 5
}

@contextmanager
def get_db_connection():
    connection = None
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        yield connection
    except Error as e:
        logger.error(f"Database connection error: {e}")
        raise
    finally:
        if connection and connection.is_connected():
            connection.close()

def insert_detection(category, timestamp):
    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            sql = "INSERT INTO trash (category, timestamp) VALUES (%s, %s)"
            cursor.execute(sql, (category, timestamp))
            connection.commit()
            logger.info(f"Inserted into DB: {category} at {timestamp}")
            return True
    except Error as e:
        logger.error(f"Error inserting into database: {e}")
        return False

def insert_sensor_reading(sensor_id, reading_value, timestamp):
    try:
        with get_db_connection() as connection:
            cursor = connection.cursor()
            sql = """
            INSERT INTO sensor (sensor_id, reading_value, timestamp)
            VALUES (%s, %s, %s)
            """
            cursor.execute(sql, (sensor_id, reading_value, timestamp))
            connection.commit()
            logger.info(f"Inserted sensor reading: Sensor {sensor_id} = {reading_value} at {timestamp}")
            return True
    except Error as e:
        logger.error(f"Error inserting sensor readings into DB: {e}")
        return False

# -------------------------------------
# Arduino Setup
# -------------------------------------
class ArduinoController:
    def __init__(self, port='/dev/ttyACM0', baud_rate=9600):
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.component_queue = queue.Queue()
        self.cooldown_tracker = defaultdict(lambda: 0)
        self.cooldown_period = 5.0
        self.active_objects = defaultdict(list)
        self.recently_detected = defaultdict(list)
        self.is_running = False
        self.thread = None

    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino reset
            logger.info("Arduino connected successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to Arduino: {e}")
            return False

    def start(self):
        if not self.arduino:
            if not self.connect():
                return False
        self.is_running = True
        self.thread = threading.Thread(target=self._communication_loop, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.is_running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.arduino:
            self.arduino.close()

    def _communication_loop(self):
        while self.is_running:
            try:
                component = self.component_queue.get(timeout=1)
                if component:
                    logger.debug(f"Sending to Arduino: {component.strip()}")
                    self.arduino.write(component.encode('ascii', errors='ignore'))

                    response_buffer = []
                    response_timeout = time.time() + 4
                    while time.time() < response_timeout:
                        if self.arduino.in_waiting > 0:
                            response = self.arduino.readline().decode('ascii', errors='ignore').strip()
                            if response:
                                logger.debug(f"Arduino response: {response}")
                                response_buffer.append(response)

                        if "DONE" in response_buffer:
                            break

                    self._process_sensor_responses(response_buffer)

            except queue.Empty:
                time.sleep(0.5)
            except Exception as e:
                logger.error(f"Arduino communication error: {e}")
                time.sleep(0.5)

    def _process_sensor_responses(self, response_buffer):
        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        for response in response_buffer:
            if "CO2 Level" in response and "NH3 Level" in response:
                co2_status = response.split('|')[0].split(':')[1].strip()
                nh3_status = response.split('|')[1].split(':')[1].strip()

                overall_status = "TOXIC" if (co2_status == "TOXIC" or nh3_status == "TOXIC") else \
                               "ABOVE NORMAL" if (co2_status == "ABOVE NORMAL" or nh3_status == "ABOVE NORMAL") else \
                               "NORMAL"

                insert_sensor_reading(3, overall_status, timestamp_str)

            elif "Container 1" in response:
                container1_fill = int(response.split(':')[1].strip().split('%')[0])
                insert_sensor_reading(1, container1_fill, timestamp_str)
                
            elif "Container 2" in response:
                container2_fill = int(response.split(':')[1].strip().split('%')[0])
                insert_sensor_reading(2, container2_fill, timestamp_str)

# -------------------------------------
# YOLOv8 Detection + Counting
# -------------------------------------
class ObjectDetector:
    def __init__(self, model_path, conf_threshold=0.7, img_size=320):
        self.original_classes = ['Biodegradable', 'Non-biodegradable', 'Recyclable']
        self.count_classes = self.original_classes.copy()
        self.model = YOLO(model_path, task='detect')
        self.conf_threshold = conf_threshold
        self.img_size = img_size
        self.counts = {cls: 0 for cls in self.count_classes}
        self.last_detected_object = None
        self.frame_clear_time = time.time()
        self.object_presence_timeout = 1.0

    def process_frame(self, frame, arduino_controller):
        current_time = time.time()
        frame_is_clear = True
        annotated_frame = frame.copy()

        # Create a mask for the detection zone
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.rectangle(mask, (rect_x1, rect_y1), (rect_x2, rect_y2), 255, -1)

        # Optimize frame for inference
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run inference with optimized parameters
        results = self.model(frame_rgb, imgsz=self.img_size, conf=self.conf_threshold, verbose=False)
        detections = results[0].boxes

        if detections and detections.xyxy is not None and len(detections.xyxy) > 0:
            for i, box in enumerate(detections.xyxy):
                cls_id = int(detections.cls[i].item())
                class_name = self.original_classes[cls_id]
                if class_name not in self.counts:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx, cy = self._get_centroid((x1, y1, x2, y2))

                # Only process detections within the detection zone
                if self._is_in_detection_zone(cx, cy):
                    frame_is_clear = False
                    if self._should_count_object(current_time):
                        self._handle_detection(class_name, cx, cy, arduino_controller)
                    self._draw_detection(annotated_frame, x1, y1, x2, y2, class_name, cx, cy)
                else:
                    # Draw detections outside zone in a different color (gray)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
                    cv2.putText(annotated_frame, f"{class_name}", (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

        if frame_is_clear and self.last_detected_object is not None:
            self.frame_clear_time = current_time
            self.last_detected_object = None

        # Draw detection zone with semi-transparent overlay
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
        cv2.addWeighted(overlay, 0.3, annotated_frame, 0.7, 0, annotated_frame)
        
        self._draw_ui(annotated_frame)
        return annotated_frame

    def _get_centroid(self, xyxy):
        x1, y1, x2, y2 = xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    def _is_in_detection_zone(self, cx, cy):
        return (rect_x1 <= cx <= rect_x2 and rect_y1 <= cy <= rect_y2)

    def _should_count_object(self, current_time):
        return (self.last_detected_object is None and 
                current_time - self.frame_clear_time > self.object_presence_timeout)

    def _handle_detection(self, class_name, cx, cy, arduino_controller):
        self.counts[class_name] += 1
        self.last_detected_object = (class_name, cx, cy)
        logger.info(f"Detected {class_name} at {datetime.now().strftime('%H:%M:%S')} - Count: {self.counts[class_name]}")

        timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insert_detection(class_name, timestamp_str)

        # Send command to Arduino
        if class_name == 'Biodegradable':
            arduino_controller.component_queue.put('BIO\n')
        elif class_name == 'Non-biodegradable':
            arduino_controller.component_queue.put('NONBIO\n')
        elif class_name == 'Recyclable':
            arduino_controller.component_queue.put('RECY\n')

        time.sleep(3)

    def _draw_detection(self, frame, x1, y1, x2, y2, class_name, cx, cy):
        # Draw detection box in blue for objects in detection zone
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"{class_name}", (x1, y1 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)

    def _draw_ui(self, frame):
        # Draw detection zone border
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 255, 0), 2)
        
        # Draw counts
        y_offset = 30
        for i, (cls, count) in enumerate(self.counts.items()):
            cv2.putText(frame, f"{cls}: {count}", (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Webcam not detected.")
        return

    # Set camera properties
    frame_w, frame_h = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_h)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    logger.info(f"Camera initialized: {actual_w}x{actual_h}")

    # Define detection zone
    global rect_x1, rect_y1, rect_x2, rect_y2
    rect_w, rect_h = 400, 350
    rect_x1 = (actual_w - rect_w) // 2
    rect_y1 = (actual_h - rect_h) // 2
    rect_x2 = rect_x1 + rect_w
    rect_y2 = rect_y1 + rect_h

    # Initialize components
    arduino_controller = ArduinoController()
    if not arduino_controller.start():
        logger.error("Failed to start Arduino controller")
        return

    detector = ObjectDetector('/home/thesis/Downloads/MGA MODELS/MODELS/AUGMENTED/v8_75ep_edgetpu.tflite')

    logger.info("Detection started. Press 'q' to quit.")

    try:
        while True:
            tic = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            annotated_frame = detector.process_frame(frame, arduino_controller)

            # Calculate and display FPS
            fps = 1.0 / (time.time() - tic)
            cv2.putText(annotated_frame, f'FPS: {fps:.2f}', 
                       (annotated_frame.shape[1] - 150, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow('Detection & Sensor Interface', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        arduino_controller.stop()

if __name__ == "__main__":
    main()
