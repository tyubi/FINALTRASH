import cv2
from ultralytics import YOLO
import serial
import time
import threading
import queue
from collections import defaultdict

# Initialize serial communication with Arduino
arduino = serial.Serial('COM5', 9600, timeout=1)  # Replace 'COM8' with your Arduino's port
time.sleep(2)  # Allow time for the connection to establish

# Load the YOLOv8 model
model = YOLO("YOLOv8sv6.pt")

# Shared queue for FIFO
component_queue = queue.Queue()

# Cooldown dictionary to track the last detection time for each component type
cooldown_tracker = defaultdict(lambda: 0)
cooldown_period = 2  # Seconds to wait before adding the same component again

# Arduino communication thread
def arduino_communication():
    while True:
        try:
            component = component_queue.get(timeout=1)  # Non-blocking get with timeout
            if component:
                print(f"Processing component: {component}")
                arduino.write(component.encode())  # Send command to Arduino

                # Wait for Arduino confirmation
                response_timeout = time.time() + 2  # 2-second timeout
                while time.time() < response_timeout:
                    if arduino.in_waiting > 0:
                        response = arduino.readline().decode().strip()
                        print(f"Arduino response: {response}")
                        if response == "DONE":
                            print(f"Component {component} processed successfully.")
                            break
                else:
                    print("Arduino response timeout. Moving to the next component.")
        except queue.Empty:
            print("Queue is empty. Waiting for new components...")
            time.sleep(0.5)

# Start Arduino communication thread
arduino_thread = threading.Thread(target=arduino_communication, daemon=True)
arduino_thread.start()

# Function to handle YOLO detection and video feed
def start_detection():
    start_time = time.time()  # Record the start time of camera initialization
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    camera_init_time = time.time() - start_time  # Calculate the initialization time
    print(f"Camera initialized in {camera_init_time:.2f} seconds.")

    prev_frame_time = 0  # Initialize for FPS calculation
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        # Run YOLO inference on the frame
        results = model(frame)
        valid_detection = False  # Track if any valid detection occurred

        # Process detection results
        for detection in results[0].boxes:
            cls = int(detection.cls)
            confidence = detection.conf

            if confidence > 0.75:  # Process only high-confidence detections
                valid_detection = True  # Mark as valid detection
                current_time = time.time()
                if current_time - cooldown_tracker[cls] > cooldown_period:
                    # Update the cooldown tracker for this component type
                    cooldown_tracker[cls] = current_time

                    # Add the detected component to the queue
                    if cls == 0:
                        component_queue.put('A')  # BJT
                        print("Status: BJT detected!")
                    elif cls == 1:
                        component_queue.put('B')  # LED
                        print("Status: LED detected!")
                    elif cls == 3:
                        component_queue.put('C')  # Capacitor
                        print("Status: Capacitor detected!")
                    elif cls in [2, 4, 6, 7, 9]:
                        component_queue.put('D')  # Defective component
                        print("Status: Defective component detected!")
                    elif cls == 8:
                        component_queue.put('E')  # Resistor
                        print("Status: Resistor detected!")
                    else:
                        component_queue.put('F')  # Unknown
                        print("Status: Unknown component detected!")

                    print(f"Queue after adding: {list(component_queue.queue)}")

        if not valid_detection:  # If no valid detection was made
            current_time = time.time()
            if current_time - cooldown_tracker['F'] > 20:
                component_queue.put('F')  # Add 'Unknown' to the queue
                cooldown_tracker['F'] = current_time
                print("Status: No detection. Added to Unknown bin.")

        # FPS calculation
        current_frame_time = time.time()
        fps = 1 / (current_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = current_frame_time

        # Display the FPS on the frame
        fps_text = f"FPS: {fps:.2f}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the annotated frame in a window
        annotated_frame = results[0].plot()
        cv2.imshow("Ecomp Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the detection
            break

        time.sleep(0.01)  # Adjust sleep time to balance performance

    cap.release()
    cv2.destroyAllWindows()

# Start detection in a separate thread
def run_detection():
    threading.Thread(target=start_detection).start()

# Start detection when the script is run
if __name__ == "__main__":
    run_detection()
