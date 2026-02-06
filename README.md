# object-detectation
Simple one file project to detect common daily objects 

# Real-Time Multi-Model Detection (Objects, Faces & Hands)

This project demonstrates **real-time computer vision** using **YOLOv8** models to detect multiple visual features simultaneously from a live webcam feed.

It combines:
- 🧍 Object detection  
- 😀 Face detection  
- ✋ Hand / wrist detection (via pose estimation)  

All results are merged into a single annotated output window.

---

## 🔥 Features

- Real-time webcam processing with OpenCV
- YOLOv8 inference using Ultralytics
- Automatic GPU (CUDA) support if available
- Object detection using COCO-trained YOLOv8
- Face detection using a YOLOv8 face model
- Hand detection via wrist keypoints from pose estimation
- Clean and readable visual overlays

---

## 🧠 Models Used

| Task | Model |
|----|----|
| Object Detection | `yolov8n.pt` |
| Pose / Hand Detection | `yolov8n-pose.pt` |
| Face Detection | `yolov8n-face.pt` |

> ⚠️ **Note:** `yolov8n-face.pt` is **not included by default** in Ultralytics. You must download or train a YOLOv8-compatible face detection model.

---

## 🛠️ Requirements

- Python **3.8+**
- Webcam
- NVIDIA GPU (optional but recommended)

### Python Dependencies

```bash
pip install ultralytics opencv-python torch

