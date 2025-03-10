# Simple-SAM2-Celery 🍃✂️

A simple and modular **FastAPI + Celery** setup for SAM2-based image segmentation. This project provides a skeleton code to run **asynchronous** and **distributed** inference using **Celery workers** and **Redis** as the message queue.

<p align="center">
  <img src="SAM2_Segmenter_Demo2.gif" alt="SAM2 Segmenter Demo">
</p>

## 🚀 Features
- **FastAPI**: Lightweight and easy-to-use web framework
- **Celery**: Distributed task queue for handling SAM2 inference asynchronously
- **Redis**: Acts as the broker and backend for Celery
- **GPU Worker Management**: Automatically spawns multiple workers across GPUs
- **Efficient Memory Management**: Clears GPU memory dynamically to prevent OOM issues
- **Modular Structure**: Easily adaptable to your own projects

## 📂 Project Structure
```
Simple-SAM2-Celery/
│── simple_api.py        # FastAPI app and Celery task definition
│── start_worker.py      # Script to start Celery workers on assigned GPUs
│── SAM2_Segmenter_Demo2.gif  # Demo GIF showcasing segmentation output
│── README.md            # Project documentation
```

## 🏃 Running the Application
### 1️⃣ Start FastAPI Server
```bash
python simple_api.py
```
This will launch the API server at `http://localhost:8000`.

### 2️⃣ Start Celery Workers
```bash
python start_worker.py
```
This will spawn multiple Celery workers across available GPUs.

## 🎯 How It Works
1. **Client sends an image segmentation request** (`/predict`).
2. **FastAPI forwards the request** to Celery as an async task.
3. **Celery worker processes the task** using **SAM2** for segmentation.
4. **Client can fetch results** via `/result/{task_id}`.

## 📝 API Endpoints
### **1. Submit a segmentation request**
```http
POST /predict
```
**Request JSON:**
```json
{
  "image_path": "path/to/image.png",
  "points": [[100, 150], [200, 250]],
  "labels": [1, 0]
}
```
**Response:**
```json
{
  "task_id": "abc123"
}
```

### **2. Retrieve the result**
```http
GET /result/{task_id}
```
**Response (if completed):**
```json
{
  "status": "completed",
  "result": { "masks": [[[0, 1, 1], [0, 0, 1]]] }
}
```

## 🔧 Customization
This is a **minimal skeleton** designed for easy modification. You can:
- **Swap SAM2** for another model (e.g., YOLO, Detectron2)
- **Modify worker behavior** (e.g., handle more concurrent tasks)
- **Adjust memory management** for different GPU settings

## 🏗️ Contributing
Feel free to fork, modify, and submit pull requests. This is just a starting point—make it yours! 🚀
