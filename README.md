# 🚀 Real-Time Object Detection using YOLOv8

A real-time object detection system built using **YOLOv8**, **OpenCV**, and **Python**. The project supports live webcam detection, video file inference, image inference, performance benchmarking, and detection recording in a modular architecture.

---

## 📌 Features

- 🎥 Real-time webcam object detection
- 📹 Video file detection
- 🖼️ Static image inference
- ⚡ YOLOv8-based high-speed detection
- 📊 Live FPS monitoring
- 📈 Model benchmarking (FPS & latency)
- 💾 Record annotated videos
- 📝 Export detection logs to CSV
- 🎯 Adjustable Confidence & IoU thresholds
- 📦 Modular and reusable code structure

---

## 🛠️ Tech Stack

- Python 3.x
- OpenCV
- Ultralytics YOLOv8
- NumPy

---

## 📂 Project Structure

```
Object_Detection-via-YOLOV8-Model/
│
├── Main.py
├── Detector.py
├── Video_source.py
├── Recorder.py
├── Infer image.py
├── Benchmark.py
├── .gitignore
└── README.md
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/Yash45-MJ/Object_Detection-via-YOLOV8-Model.git
```

Move into the project folder:

```bash
cd Object_Detection-via-YOLOV8-Model
```

Install dependencies:

```bash
pip install ultralytics opencv-python numpy
```

---

## ▶️ Usage

### Run Real-Time Detection

```bash
python Main.py
```

### Run on a Video

```bash
python Main.py --source video.mp4
```

### Run on an Image

```bash
python "Infer image.py" image.jpg
```

### Benchmark YOLO Models

```bash
python Benchmark.py
```

---

## 📈 Performance Features

- Letterbox preprocessing
- Adjustable confidence threshold
- IoU threshold tuning
- FPS calculation
- Benchmarking with latency statistics
- Frame skipping for higher FPS
- Detection statistics
- Recording and CSV logging

---

## 📌 Future Improvements

- ✅ Object Tracking (ByteTrack / DeepSORT)
- ✅ ONNX Runtime Support
- ✅ TensorRT Optimization
- ✅ Docker Support
- ✅ FastAPI Deployment
- ✅ Custom Dataset Training
- ✅ GPU Auto Detection
- ✅ Multi-camera Support

---

## 👨‍💻 Author

**Yash Bhite**

Final Year Student  
Robotics & Automation Engineering

### Areas of Interest

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Robotics
- Autonomous Systems

---

## ⭐ If you found this project useful, consider giving it a Star!
