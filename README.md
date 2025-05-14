
# EcompDetection

**Automated Sorting and Defect Detection of Electronic Components Using Computer Vision for PUP CPE Laboratory**

EcompDetection is a computer vision-based system designed to assist in the inspection, classification, and sorting of electronic components using a YOLOv8 model. This project is developed for the PUP Computer Engineering Laboratory to aid in the educational and quality assurance processes.

## 🔍 Features

- **Object Detection:** Identifies various electronic components and defects using a custom-trained YOLOv8 model.
- **Edge TPU Optimization:** Includes TensorFlow Lite models compiled for Coral Edge TPU to accelerate inference.
- **GUI Application:** A full-screen Tkinter-based GUI for real-time detection, object counting, and serial communication with Arduino.
- **Automated Sorting:** Interfaces with a conveyor or robotic arm system for physical sorting based on detection results.
- **Logging and Analysis:** Captures detection results and logs them for further analysis.

## 📁 Project Structure

```
EcompDetection/
│
├── ecomp_ard/                    # Arduino-related code and hardware interface
├── models/                       # Model files and saved weights
├── py files/                     # Supporting Python scripts and utilities
├── servo-test/                   # Scripts for testing servo motor movements
├── shelf/                        # Optional folder for placing electronics to be detected
├── tflite new/                   # Updated TFLite models (EdgeTPU, etc.)
│
├── ecomp-detect-yolov8n-v1_edgetpu.tflite  # EdgeTPU model (optimized)
├── ecomp-detect-yolov8n-v1.tflite          # Base quantized TFLite model
│
├── inference_ard.py              # Inference with Arduino serial communication
├── inference-gui.py              # Fullscreen GUI application
├── inference.py                  # Inference only (no GUI or serial)
├── README.md                     # Project documentation
```

## 🖥️ Requirements

- Python 3.9.12
- OpenCV 4.5.5.62
- PyCoral
- Tkinter
- Ultralytics 8.2.73
- Serial Communication (pyserial)
- Edge TPU Runtime (for Coral USB Accelerator)
- Arduino IDE (for firmware)

## 🚀 Running the Application

📌 **Note:** It is recommended to create and activate a virtual environment before running the application to avoid dependency conflicts.

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

**Inference test:**
```bash
python inference.py
```

**With Arduino communication enabled:**
```bash
python inference_ard.py
```

**For viewing the whole project with GUI:**
```bash
python inference-gui.py
```

## 🤖 Hardware Integration

- **Camera:** USB webcam (mounted behind or above robotic arm)
- **Processing:** Raspberry Pi 4B 8GB with Coral USB Accelerator
- **Actuator:** Servo Motor (controlled via Arduino UNO)
- **Components:** Resistors, capacitors, LEDs, and defective parts (e.g., rusted, cracked, missing leg)

## 📌 Status

- [/] Model trained and converted
- [/] Real-time GUI developed
- [/] Edge TPU tested successfully
- [/] Arduino integration done
- [ ] Final deployment and enclosure

## 📜

This project is developed for academic purposes at Polytechnic University of the Philippines and is open for educational and research use.

---

**Developed by BSCpE - Group4202**
