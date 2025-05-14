import cv2
import numpy as np
import time
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from threading import Thread

# Class for threaded video capture
class VideoStream:
    def __init__(self, src=0, resolution=(640, 480)):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True
        self.stream.release()

# Main function
def main():
    # Load the TFLite model
    print("Loading model on Edge TPU...")
    interpreter = edgetpu.make_interpreter("ecomp_detect_yolov8n_int8.tflite")
    interpreter.allocate_tensors()
    
    # Get model details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Print model information
    print(f"Input details: {input_details}")
    print(f"Output details: {output_details}")
    
    # Get input size from model
    input_shape = input_details[0]['shape']
    input_height = input_shape[1]
    input_width = input_shape[2]
    input_size = (input_width, input_height)
    print(f"Model input size: {input_size}")
    
    # Check input quantization parameters
    input_scale, input_zero_point = input_details[0]['quantization']
    print(f"Input quantization: scale={input_scale}, zero_point={input_zero_point}")
    
    # Resolution for video capture
    resolution = (480, 480)
    
    # Initialize video stream
    print("Starting video capture...")
    vs = VideoStream(src=0, resolution=resolution).start()
    time.sleep(1.0)  # Allow camera to warm up
    
    # Define class names (update with your actual classes)
    class_names = ["person", "bicycle", "car"]
    
    # Define confidence threshold
    conf_threshold = 0.5
    
    # For FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    print("Press 'q' to quit...")
    try:
        while True:
            # Get frame
            frame = vs.read()
            if frame is None:
                continue
            
            # Resize to match model input size
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(frame_rgb, input_size)
            
            # Prepare input data - using the correct method for TPU
            if input_details[0]['dtype'] == np.uint8:
                # For quantized model (uint8)
                input_data = np.expand_dims(resized, axis=0)
            else:
                # For floating point model
                input_data = (np.expand_dims(resized, axis=0) / 255.0).astype(np.float32)
            
            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], input_data)
            
            # Run inference - measure time
            inference_start = time.time()
            interpreter.invoke()
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Get output tensor
            output_data = interpreter.get_tensor(output_details[0]['index'])
            
            # Process detections
            height, width = frame.shape[:2]
            boxes = []
            confidences = []
            class_ids = []
            
            # Determine number of classes from output shape
            num_classes = output_data.shape[2] - 5
            
            # Process detections
            for detection in output_data[0]:
                confidence = detection[4]
                
                if confidence > conf_threshold:
                    class_scores = detection[5:5+num_classes]
                    class_id = np.argmax(class_scores)
                    class_confidence = class_scores[class_id]
                    
                    if class_confidence > conf_threshold:
                        # YOLOv8 outputs are normalized [0, 1]
                        cx, cy, w, h = detection[0:4]
                        
                        # Convert to corner format (xmin, ymin, xmax, ymax)
                        x1 = int((cx - w/2) * width)
                        y1 = int((cy - h/2) * height)
                        x2 = int((cx + w/2) * width)
                        y2 = int((cy + h/2) * height)
                        
                        # Keep boxes within image boundaries
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(width, x2), min(height, y2)
                        
                        boxes.append([x1, y1, x2, y2])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # Draw boxes on frame (simplified for better performance)
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                # Draw bounding box (simplified)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                
                # Only draw labels if we have enough boxes
                if len(boxes) < 10:  # Skip detailed labels if there are too many objects
                    class_id = class_ids[i]
                    conf = confidences[i]
                    
                    # Get class name
                    if class_id < len(class_names):
                        label = f"{class_names[class_id]}: {conf:.2f}"
                    else:
                        label = f"Class {class_id}: {conf:.2f}"
                    
                    # Add label (simplified)
                    cv2.putText(frame, label, (x1, y1-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Update FPS calculation
            fps_counter += 1
            if (time.time() - fps_start_time) > 1.0:  # Update every second
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Display performance info
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(frame, f"Inference: {inference_time:.1f}ms", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display the frame
            cv2.imshow('Edge TPU YOLOv8 Detection', frame)
            
            # Check for keypress
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        vs.stop()
        cv2.destroyAllWindows()
        print("Resources released")

if __name__ == "__main__":
    main()
