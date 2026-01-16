# Face Detection System using Haar Cascades and Deep Learning

## 📌 Project Overview

This project implements a **Face Detection System** capable of detecting human faces in **images and real-time video streams**. The system uses a **hybrid approach**:

* **Haar Cascade Classifier** for fast, real-time face detection
* **Deep Learning–based SSD (Single Shot Detector)** model for improved accuracy in challenging conditions

The project is developed using **Python**, **OpenCV**, and **TensorFlow**, following industry-standard practices suitable for academic projects, internships, and beginner-to-intermediate AI/ML learners.

---

## 🎯 Objectives

* Detect faces in static images
* Detect faces in live webcam/video feeds
* Compare classical ML and deep learning approaches
* Log detected faces automatically for analysis
* Build a modular and scalable face detection pipeline

---

## 🛠️ Technologies Used

* **Python 3.x**
* **OpenCV** – image and video processing
* **TensorFlow** – deep learning support
* **NumPy** – numerical operations

---

## 📁 Project Structure

```
FACE_DETECTION_SYSTEM/
│
├── myenv/                                 # Virtual environment
├── image_face_detect.py                  # Haar cascade – image detection
├── video_face_detect.py                  # Haar cascade – video/webcam detection
├── deep_learning_face.py                 # Deep learning face detection
├── haarcascade_frontalface_default.xml   # Haar cascade classifier
├── deploy.prototxt                       # DL model architecture
├── res10_300x300_ssd_iter_140000.caffemodel # DL model weights
├── requirements.txt                      # Project dependencies
├── face_log.txt                          # Auto-generated face detection log
└── README.md                             # Project documentation
```

---

## ⚙️ Setup Instructions

### 1️⃣ Clone or Download the Project

Place the project folder on your local machine.

---

### 2️⃣ Create & Activate Virtual Environment

**Windows:**

```bash
python -m venv myenv
myenv\Scripts\activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing:

```bash
pip install opencv-python numpy tensorflow matplotlib
```

---

## ▶️ How to Run the Project

### 🔹 Image Face Detection (Haar Cascade)

1. Place a test image (e.g., `test.jpg`) in the project root
2. Run:

```bash
python image_face_detect.py
```

---

### 🔹 Video / Webcam Face Detection (Haar Cascade)

```bash
python video_face_detect.py
```

Press **Q** to exit the webcam window.

---

### 🔹 Deep Learning Face Detection

```bash
python deep_learning_face.py
```

This uses a pre-trained SSD + ResNet model for more accurate detection.

---

## 📝 Face Logging

* The file `face_log.txt` is created automatically
* Each detected face is logged with a timestamp

**Sample log:**

```
Face detected at 2026-01-16 19:05:12
```

---

## 📊 Comparison of Approaches

| Method              | Speed     | Accuracy | Use Case             |
| ------------------- | --------- | -------- | -------------------- |
| Haar Cascade        | Very Fast | Moderate | Real-time systems    |
| Deep Learning (SSD) | Moderate  | High     | Complex environments |

---

## 🚀 Applications

* Security & surveillance systems
* Attendance systems
* Smart cameras
* Retail analytics
* Human–computer interaction

---

## ⚠️ Known Limitations

* Haar cascades may fail in poor lighting or angled faces
* Deep learning model requires more computational resources

---

## 🔮 Future Enhancements

* Face recognition (identity matching)
* Mask detection
* Emotion detection
* Web or mobile deployment
* GPU acceleration

---

## 👨‍💻 Author

Developed as a learning and academic project to understand **computer vision and face detection techniques** using both traditional ML and deep learning.

---

## 📜 License

This project is for **educational and learning purposes only**.
