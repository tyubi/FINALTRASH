import tkinter
import customtkinter
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import queue
from collections import deque
from datetime import datetime
from ultralytics import YOLO

customtkinter.set_appearance_mode("System")
customtkinter.set_default_color_theme("blue")

class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        
        # Initialize image attribute to prevent garbage collection
        self.current_image = None
        
        # Frame size for better performance on Raspberry Pi
        self.frame_width = 320
        self.frame_height = 240
        
        # Create a frame buffer queue - increased size to reduce frame drops
        self.frame_queue = queue.Queue(maxsize=5)
        
        # Start frame consumer for smoother display
        self.frame_consumer_running = False
        
        # IMPORTANT: Use a single frame counter to keep display and camera in sync
        self.frame_counter = 0
        self.target_fps = 15  # Set target FPS for both camera and display
        self.last_frame_time = 0
        
        # FPS tracking variables - now using a single value for display
        self.current_fps = 0
        self.fps_counter = 0
        self.fps_start_time = time.time()
        
        # YOLOv8 Detection settings
        self.class_names = ['BJT', 'LED', 'burnt', 'capacitor', 'cracked', 'defective', 'faded', 'missing-leg', 'resistor', 'rust']
        self.counts = {cls: 0 for cls in self.class_names}
        self.seen_centroids = {cls: [] for cls in self.class_names}
        self.line_x = 150  # X-coordinate for vertical line
        self.offset = 10  # Line crossing tolerance
        self.min_distance = 30  # Minimum centroid distance
        self.centroid_timeout = 1.0  # Time before a centroid can be counted again
        self.model = None  # Will load when starting detection
        self.detection_running = False
        
        # Additional optimization settings for Raspberry Pi
        self.detection_size = 256
        self.confidence_threshold = 0.5
        
        # configure window
        self.title("EcompDetection")
        self.attributes("-fullscreen", True)
        self.bind("<Escape>", lambda event: self.on_closing())

        # configure grid layout (3 columns)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # left sidebar
        self.sidebar_frame = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=2, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(1, weight=1)
        self.logo_label = customtkinter.CTkLabel(
            self.sidebar_frame, text="EcompDetection", font=customtkinter.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))
        spacer = customtkinter.CTkLabel(self.sidebar_frame, text="")
        spacer.grid(row=1, column=0, sticky="nsew")

        # appearance and scaling
        self.appearance_mode_label = customtkinter.CTkLabel(self.sidebar_frame, text="Appearance Mode:", anchor="w")
        self.appearance_mode_label.grid(row=2, column=0, padx=20, pady=(10, 0), sticky="w")
        self.appearance_mode_optionemenu = customtkinter.CTkOptionMenu(
            self.sidebar_frame, values=["Light", "Dark", "System"], command=self.change_appearance_mode_event
        )
        self.appearance_mode_optionemenu.grid(row=3, column=0, padx=20, pady=(5, 10), sticky="w")

        self.scaling_label = customtkinter.CTkLabel(self.sidebar_frame, text="UI Scaling:", anchor="w")
        self.scaling_label.grid(row=4, column=0, padx=20, pady=(10, 0), sticky="w")
        self.scaling_optionemenu = customtkinter.CTkOptionMenu(
            self.sidebar_frame,
            values=["80%", "90%", "100%", "110%", "120%", "200%"],
            command=self.change_scaling_event
        )
        self.scaling_optionemenu.grid(row=5, column=0, padx=20, pady=(5, 20), sticky="w")

        # Add performance options
        self.perf_label = customtkinter.CTkLabel(self.sidebar_frame, text="Performance:", anchor="w")
        self.perf_label.grid(row=6, column=0, padx=20, pady=(10, 0), sticky="w")
        
        # FPS slider replaces frame skip slider
        self.fps_slider_label = customtkinter.CTkLabel(self.sidebar_frame, text=f"Target FPS: {self.target_fps}", anchor="w")
        self.fps_slider_label.grid(row=7, column=0, padx=20, pady=(5, 0), sticky="w")
        self.fps_slider = customtkinter.CTkSlider(
            self.sidebar_frame, from_=5, to=30, number_of_steps=25, command=self.update_target_fps
        )
        self.fps_slider.grid(row=8, column=0, padx=20, pady=(5, 10), sticky="w")
        self.fps_slider.set(self.target_fps)
        
        # Single FPS Display
        self.fps_label = customtkinter.CTkLabel(self.sidebar_frame, text="FPS Stats:", anchor="w")
        self.fps_label.grid(row=9, column=0, padx=20, pady=(10, 0), sticky="w")
        self.current_fps_label = customtkinter.CTkLabel(self.sidebar_frame, text="Current: 0.0 FPS", anchor="w")
        self.current_fps_label.grid(row=10, column=0, padx=20, pady=(5, 10), sticky="w")

        # middle column configuration with video frame
        self.grid_columnconfigure(1, weight=1)
        self.middle_frame = customtkinter.CTkFrame(self)
        self.middle_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")
        self.middle_frame.grid_rowconfigure(0, weight=1)
        self.middle_frame.grid_columnconfigure(0, weight=1)
        
        # Create a label for displaying video with a background color and initial text
        self.video_label = customtkinter.CTkLabel(
            self.middle_frame, 
            text="Camera Feed (Click 'Start Detection')",
            fg_color=("gray75", "gray25"),
            corner_radius=8
        )
        self.video_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        
        # Video capture variables
        self.cap = None
        self.is_capturing = False
        self.video_thread = None

        # middle textbox for logs
        self.textbox = customtkinter.CTkTextbox(self, height=100)
        self.textbox.grid(row=1, column=1, padx=20, pady=20, sticky="ew")
        self.textbox.insert("0.0", "Detection Log\n\n")
        self.textbox.configure(wrap="word")

        # right sidebar
        self.right_sidebar = customtkinter.CTkFrame(self, width=140, corner_radius=0)
        self.right_sidebar.grid(row=0, column=2, rowspan=2, sticky="nsew")
        self.right_sidebar.grid_rowconfigure((2, 3), weight=1)

        self.button_1 = customtkinter.CTkButton(
            self.right_sidebar, text="Start Detection", command=self.start_detection
        )
        self.button_1.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="n")
        self.button_2 = customtkinter.CTkButton(
            self.right_sidebar, text="Stop Detection", command=self.stop_detection, state="disabled"
        )
        self.button_2.grid(row=1, column=0, padx=20, pady=10, sticky="n")
        
        # Reset counts button
        self.reset_button = customtkinter.CTkButton(
            self.right_sidebar, text="Reset Counts", command=self.reset_counts
        )
        self.reset_button.grid(row=2, column=0, padx=20, pady=10, sticky="n")

        # default settings
        self.appearance_mode_optionemenu.set("Dark")
        self.scaling_optionemenu.set("100%")
        customtkinter.set_widget_scaling(1.0)
        
        # Clean up on window close
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # FPS update timer
        self.after(1000, self.update_fps_display)
        
        # System info display at startup
        self.log_message("System: Raspberry Pi 4B")
        self.log_message(f"Resolution: {self.frame_width}x{self.frame_height}")
        self.log_message(f"Target FPS: {self.target_fps}")

    def update_fps_display(self):
        """Update the FPS display label"""
        if self.is_capturing:
            self.current_fps_label.configure(text=f"Current: {self.current_fps:.1f} FPS")
        self.after(1000, self.update_fps_display)

    def update_target_fps(self, value):
        """Update the target FPS for synchronized capture and display"""
        self.target_fps = int(value)
        self.fps_slider_label.configure(text=f"Target FPS: {self.target_fps}")
        self.log_message(f"Target FPS set to: {self.target_fps}")
        
        # Reset the FPS counter when changing settings
        self.fps_counter = 0
        self.fps_start_time = time.time()

    def change_appearance_mode_event(self, new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    def change_scaling_event(self, new_scaling: str):
        new_scaling_float = int(new_scaling.replace("%", "")) / 100
        customtkinter.set_widget_scaling(new_scaling_float)
    
    def reset_counts(self):
        """Reset all detection counts"""
        self.counts = {cls: 0 for cls in self.class_names}
        self.seen_centroids = {cls: [] for cls in self.class_names}
        self.log_message("Detection counts reset to zero")
        
    def get_centroid(self, xyxy):
        """Calculate centroid from bounding box coordinates"""
        x1, y1, x2, y2 = xyxy
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    
    def log_message(self, message):
        """Add message to log with timestamp"""
        # Use after() to ensure thread safety when updating UI
        self.after(0, lambda: self._update_log(message))
    
    def _update_log(self, message):
        """Actually update the log textbox (called from main thread)"""
        self.textbox.insert("end", f"{time.strftime('%H:%M:%S')}: {message}\n")
        self.textbox.see("end")  # Auto-scroll to the end
    
    def frame_consumer(self):
        """Consume frames from the queue, process with YOLOv8, and update the UI"""
        frame_interval = 1.0 / self.target_fps  # Time between frames for target FPS
        last_frame_time = time.time()
        
        while self.frame_consumer_running:
            try:
                # Calculate time until next frame should be processed
                current_time = time.time()
                elapsed = current_time - last_frame_time
                
                # If it's not time for the next frame yet, wait
                if elapsed < frame_interval:
                    time.sleep(0.001)  # Short sleep to avoid CPU hogging
                    continue
                
                # Time to process a new frame - reset timer
                last_frame_time = current_time
                
                # Get frame from queue with timeout
                try:
                    frame = self.frame_queue.get(timeout=frame_interval)
                except queue.Empty:
                    continue
                
                # Update FPS counter
                self.fps_counter += 1
                fps_elapsed = current_time - self.fps_start_time
                if fps_elapsed >= 1.0:  # Update FPS every second
                    self.current_fps = self.fps_counter / fps_elapsed
                    self.fps_counter = 0
                    self.fps_start_time = current_time
                
                # Process with YOLOv8 if detection is enabled
                if self.detection_running and self.model is not None:
                    try:
                        result = self.model(frame, imgsz=self.detection_size, conf=self.confidence_threshold, verbose=False)
                        detections = result[0].boxes
                        annotated_frame = result[0].plot()
                    except Exception as e:
                        self.log_message(f"Detection error: {str(e)}")
                        annotated_frame = frame.copy()
                    
                    # Clean up old centroids
                    current_time = time.time()
                    for cls in self.class_names:
                        # Keep only centroids that are less than centroid_timeout seconds old
                        self.seen_centroids[cls] = [
                            (cx, cy, t) for cx, cy, t in self.seen_centroids[cls] 
                            if current_time - t < self.centroid_timeout
                        ]
                    
                    # Process detections
                    if detections is not None and len(detections) > 0 and detections.xyxy is not None:
                        for i, box in enumerate(detections.xyxy):
                            cls_id = int(detections.cls[i].item())
                            class_name = self.class_names[cls_id]
                            conf = float(detections.conf[i].item())
                            x1, y1, x2, y2 = map(int, box)
                            cx, cy = self.get_centroid((x1, y1, x2, y2))
                            
                            # Draw centroid
                            cv2.circle(annotated_frame, (cx, cy), 3, (0, 255, 255), -1)
                            
                            # Check if object crosses vertical line
                            if abs(cx - self.line_x) < self.offset:
                                already_counted = False
                                
                                # Check if this centroid has been counted recently
                                for prev_cx, prev_cy, prev_time in self.seen_centroids[class_name]:
                                    distance = np.sqrt((cx - prev_cx) ** 2 + (cy - prev_cy) ** 2)
                                    if distance < self.min_distance:
                                        already_counted = True
                                        break
                                
                                if not already_counted:
                                    self.counts[class_name] += 1
                                    self.seen_centroids[class_name].append((cx, cy, current_time))
                                    
                                    # Log the detection
                                    detection_msg = f"Detected {class_name} - Count: {self.counts[class_name]}"
                                    self.log_message(detection_msg)
                    
                    # Draw vertical counting line
                    cv2.line(annotated_frame, (self.line_x, 0), (self.line_x, annotated_frame.shape[0]), 
                            (255, 0, 0), 2)
                    
                    # Show counts - more compact display for small screen
                    y_offset = 20
                    for i, (cls, count) in enumerate(self.counts.items()):
                        # Display in two columns for better visibility
                        col = 0 if i < 5 else 1
                        row = i % 5
                        x_pos = 10 if col == 0 else self.frame_width // 2
                        cv2.putText(annotated_frame, f"{cls}: {count}", (x_pos, y_offset + row * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    
                    # Show FPS
                    cv2.putText(annotated_frame, f'FPS: {self.current_fps:.1f}', 
                               (annotated_frame.shape[1] - 100, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                    
                    # Use the annotated frame with detections
                    display_frame = annotated_frame
                else:
                    # Just use the original frame if detection is off
                    display_frame = frame
                    
                    # Add FPS counter to the raw frame too
                    cv2.putText(display_frame, f'FPS: {self.current_fps:.1f}', 
                               (display_frame.shape[1] - 100, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Convert to RGB for PIL
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                # Get current dimensions of the video label
                width = self.video_label.winfo_width()
                height = self.video_label.winfo_height()
                
                # If widget hasn't been drawn yet or has invalid size
                if width <= 1 or height <= 1:
                    width = self.frame_width
                    height = self.frame_height
                
                # Create and store the CTkImage
                self.current_image = customtkinter.CTkImage(
                    light_image=pil_image,
                    dark_image=pil_image,
                    size=(width, height)
                )
                
                # Update the UI in the main thread
                self.after(1, lambda: self.video_label.configure(image=self.current_image, text=""))
                
                # Mark frame as processed
                self.frame_queue.task_done()
                
            except Exception as e:
                self.log_message(f"Error in frame processing: {str(e)}")
                time.sleep(0.1)

    def start_detection(self):
        if not self.is_capturing:
            self.is_capturing = True
            self.log_message("Loading YOLOv8 model...")
            
            try:
                # Load YOLOv8 model if not already loaded
                if self.model is None:
                    # Set optimization flags for TFLite on Raspberry Pi
                    self.model = YOLO('ecomp-detect-yolov8n_edgetpu.tflite', task='detect')
                    self.log_message("YOLOv8 model loaded successfully")
                
                self.detection_running = True
                self.log_message("Starting detection...")
                
                # Clear existing frame queue
                while not self.frame_queue.empty():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                # Reset FPS counter
                self.fps_counter = 0
                self.fps_start_time = time.time()
                
                # Start frame consumer thread
                self.frame_consumer_running = True
                self.frame_consumer_thread = threading.Thread(target=self.frame_consumer)
                self.frame_consumer_thread.daemon = True
                self.frame_consumer_thread.start()
                
                # Start video capture in a separate thread
                self.video_thread = threading.Thread(target=self.video_capture_loop)
                self.video_thread.daemon = True
                self.video_thread.start()
                
                self.button_1.configure(state="disabled")
                self.button_2.configure(state="normal")
            except Exception as e:
                self.log_message(f"Error starting detection: {str(e)}")
                self.is_capturing = False

    def stop_detection(self):
        if self.is_capturing:
            self.is_capturing = False
            self.detection_running = False
            self.frame_consumer_running = False
            self.log_message("Stopping detection...")
            
            if self.video_thread:
                self.video_thread.join(timeout=1.0)
            
            if hasattr(self, 'frame_consumer_thread'):
                self.frame_consumer_thread.join(timeout=1.0)
            
            if self.cap and self.cap.isOpened():
                self.cap.release()
                self.cap = None
            
            # Clear the video display
            self.video_label.configure(image=None, text="Camera Feed Stopped")
            
            # Reset FPS display
            self.current_fps_label.configure(text="Current: 0.0 FPS")
            
            self.button_1.configure(state="normal")
            self.button_2.configure(state="disabled")

    def video_capture_loop(self):
        """Capture frames from camera at the target FPS rate"""
        cv2.setUseOptimized(True)
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            self.log_message("Error: Could not open camera")
            self.is_capturing = False
            return
        
        # Set camera properties for better performance if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        self.log_message("Camera initialized")
        
        # Use the same frame interval as the consumer
        frame_interval = 1.0 / self.target_fps
        last_capture_time = time.time()
        
        while self.is_capturing:
            current_time = time.time()
            elapsed = current_time - last_capture_time
            
            # If it's not time for the next frame, wait a bit
            if elapsed < frame_interval:
                time.sleep(0.001)  # Short sleep
                continue
                
            # Reset timer for next frame
            last_capture_time = current_time
            
            # Capture frame
            ret, frame = self.cap.read()
            
            if not ret:
                self.log_message("Error: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            # Put frame in queue, skip if queue is full
            try:
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Skip frame if queue is full
        
        if self.cap and self.cap.isOpened():
            self.cap.release()

    def on_closing(self):
        self.stop_detection()
        self.destroy()

if __name__ == "__main__":
    app = App()
    app.mainloop()