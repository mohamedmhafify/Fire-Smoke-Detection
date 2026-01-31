# 🔥 Fire & Smoke Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLO-Ultralytics-orange?style=for-the-badge&logo=yolo&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)

## 📖 Project Overview

The **Fire & Smoke Detection System** is a cutting-edge computer vision solution designed to detect fire and smoke in real-time. Built using the **YOLO (You Only Look Once)** architecture, this system processes images and video streams to provide early warnings for potential hazards.

This project aims to enhance safety in various environments by leveraging deep learning to identify visual patterns of fire and smoke, which are often dynamic and difficult to detect with traditional sensors.

### 🎯 Use Cases
- **Surveillance Systems:** Automated monitoring of CCTV feeds.
- **Industrial Safety:** Early detection in factories and warehouses.
- **Smart Buildings:** Integration with building management systems.
- **Forestry:** Wildfire detection from drone or tower feeds.

---

## ✨ Key Features

- **Real-Time Detection:** High-speed inference suitable for live video feeds.
- **Dual Class Detection:** Accurately distinguishes between **Fire** and **Smoke**.
- **Interactive Dashboard:** User-friendly interface built with **Streamlit**.
- **Multi-Source Support:** Upload images or process video files (`.mp4`, `.avi`, `.mov`).
- **Performance Metrics:** Built-in dashboard to view precision, recall, and mAP scores.

---

## 🛠️ Tech Stack

- **Language:** Python
- **Model:** YOLO (Ultralytics)
- **Computer Vision:** OpenCV, Pillow
- **Data Manipulation:** NumPy, Pandas
- **Interface:** Streamlit

---

## 📂 Project Structure

```text
Fire-Smoke-Detection/
├── assets/              # Images, videos, and demo resources
├── notebooks/           # Jupyter notebooks for training and experiments
├── app.py               # Main Streamlit application entry point
├── best.pt              # Trained YOLO model weights
├── data.yaml            # Dataset configuration file
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore rules
└── README.md            # Project documentation
```

---

## 🚀 Installation

Follow these steps to set up the project locally.

### 1. Clone the Repository
```bash
git clone https://github.com/mohamedmhafify/fire-smoke-detection.git
cd fire-smoke-detection
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 💻 Usage

This project uses **Streamlit** for the user interface. To run the application:

```bash
streamlit run app.py
```

Once the server starts, open your browser (usually at `http://localhost:8501`) to interact with the system.

### Application Modes:
1.  **Detection Demo:** Upload an image or video to see the model in action.
2.  **Performance Dashboard:** View confusion matrices, training curves, and F1 scores.
3.  **Training Story:** Insights into how the model was trained and optimized.

---

## 📊 Dataset Details

The model was trained on a custom dataset specifically curated for fire and smoke scenarios.

- **Configuration:** Defined in `data.yaml`.
- **Classes:**
  - `0`: Fire
  - `1`: Smoke
- **Format:** YOLO format (txt annotations).

---

## 🧠 Model Details

- **Architecture:** YOLO (Ultralytics)
- **Weights:** `best.pt` (included in root directory)
- **Optimization:** Tuned for a balance between inference speed (latency) and detection accuracy (mAP).
- **Training:** Trained over 100 epochs with specific augmentations to handle the non-rigid nature of smoke.

---

## 📸 Screenshots

> *Placeholder for screenshots of the Streamlit interface, detection results on images, and the performance dashboard.*

---

## 🔮 Future Improvements / Roadmap

- [ ] Integration with live IP Camera (RTSP) feeds.
- [ ] Email/SMS alert system upon detection.
- [ ] Deployment to edge devices (Jetson Nano / Raspberry Pi).
- [ ] Night vision optimization.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👨‍💻 Author

**Mohamed Mostafa Hassan Afify**
- GitHub
- LinkedIn