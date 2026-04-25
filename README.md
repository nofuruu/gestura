<div align="center">
  <img src="src/gestura/assets/gestura-no_background.png" alt="Gestura Engine Logo" width="250">
  
  # Gestura Engine v 1.0.0
  
  **Real-time Hand Gesture Recognition & Expert System**
</div>

A computer vision-based gesture recognition system designed to detect and classify hand signs (Sign Language) using 21 hand landmarks. Gestura integrates **MediaPipe** for feature extraction and a custom **K-Nearest Neighbors (KNN)** algorithm for real-time classification with optimized performance.

---

## ✨ Features

- **Real-time Inference Engine**: Detect and classify gestures with low latency using NumPy optimizations
- **Professional Technical Dashboard**: Built with DearPyGui for live monitoring of hand landmarks and model performance
- **Dynamic Dataset Management**: SQLite integration for storing and managing feature vectors without relying on static CSV files
- **Live Analytics Panel**: Monitor confidence scores and coordinate matrices in real-time
- **Modular Architecture**: Clean separation of concerns between UI, inference logic, and data management
- **Hardware Optimization**: Efficient NumPy-based KNN implementation for instant predictions
- **Configurable Parameters**: Dynamically adjust K-neighbors and confidence thresholds without restarting

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.10+ |
| **Computer Vision** | OpenCV, MediaPipe |
| **Inference Logic** | NumPy (Custom KNN) |
| **GUI** | DearPyGui (GPU Accelerated) |
| **Database** | SQLite3 |
| **Data Processing** | Pandas, NumPy |

---

## 📂 Project Structure

```
Gestura/
├── main.ipynb                    # Research, preprocessing & model experimentation
├── gestura.py                    # Entry point (Main UI & render loop)
├── engine.py                     # Core KNN logic & normalization
├── database_manager.py           # Data layer (SQLite CRUD operations)
├── dataset/
│   ├── Datafull terakhir test.csv
│   └── Datafull train.csv
├── example/
└── README.md
```

---

## 🚀 Installation

### Prerequisites
- Python 3.10 or higher
- Webcam/camera device
- ~500MB disk space for dependencies

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/nofuruu/gestura.git
   cd gestura
   ```

2. **Create virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   
   **Option A: Using requirements.txt (Recommended)**
   
   The `requirements.txt` file contains all necessary Python packages with their compatible versions. Install them with a single command:
   ```bash
   pip install -r requirements.txt
   ```
   
   This will automatically install all dependencies listed in the file, including:
   - opencv-python
   - mediapipe
   - dearpygui
   - numpy
   - pandas
   
   **Option B: Manual installation**
   
   If you prefer to install packages individually:
   ```bash
   pip install opencv-python mediapipe dearpygui numpy pandas
   ```

---

## 🎮 Quick Start

### Running the Application

```bash
python gestura.py
```

The application will launch with a professional dashboard featuring:
- Live camera feed with hand landmarks visualization
- Real-time gesture classification
- Performance metrics and logging

### User Guide

1. **Initialize Engine**: Click the `START ENGINE` button to initialize the camera and load the KNN model into memory
2. **Capture Gesture Data**: Press `C` key to capture the current hand landmark coordinates
3. **Adjust Configuration**: Use sliders in the left panel to dynamically change K-neighbors or confidence threshold without restarting
4. **View Logs**: Monitor live inference logs including confidence scores and latency metrics
5. **Terminate**: Click `TERMINATE` to safely shutdown hardware and save all logs to the database

---

## 🧠 Algorithm Details

### K-Nearest Neighbors (KNN) Classification

Gestura uses **Euclidean distance** to determine similarity between input coordinates and the training dataset:

$$d(x, y) = \sqrt{\sum_{i=1}^{n} (x_i - y_i)^2}$$

### Normalization

All hand coordinates are normalized relative to the wrist point (Landmark 0) to ensure consistent accuracy regardless of hand distance from the camera.

### Feature Extraction

- **Input**: 21 hand landmarks extracted by MediaPipe (each with x, y coordinates)
- **Features**: 42-dimensional vector (21 points × 2 coordinates)
- **Preprocessing**: Normalized against wrist position
- **Output**: Predicted gesture class with confidence score

---

## 📊 Data Management

### Database Schema

The system maintains three SQLite tables:

1. **hand_dataset**: Training data with 21 hand landmark coordinates and gesture labels
2. **inference_logs**: Timestamped prediction results with confidence and latency metrics
3. **settings**: Configuration parameters and system settings

### Training Data Format

```csv
label, point_0x, point_0y, point_1x, point_1y, ..., point_20x, point_20y
A,     0.5,     0.3,     0.48,    0.25,    ...,  0.52,     0.35
B,     0.51,    0.32,    0.49,    0.26,    ...,  0.53,     0.36
```

---

## 🔧 Configuration

Edit model parameters in the UI or directly in `engine.py`:

- **K-neighbors**: Number of nearest neighbors to consider (default: 3)
- **Confidence Threshold**: Minimum confidence for valid predictions
- **Detection Confidence**: MediaPipe hand detection threshold (default: 0.7)
- **Tracking Confidence**: MediaPipe tracking confidence (default: 0.7)

---

## 📈 Performance Metrics

The system tracks:
- **Inference Latency**: Time taken for gesture classification (milliseconds)
- **Confidence Score**: Probability of the predicted gesture (0-1)
- **Detection Rate**: Success rate of hand landmark detection
- **Memory Usage**: Real-time resource monitoring

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

### Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Project Goals

This project serves as a research implementation for:
- Real-time gesture recognition systems
- Computer vision applications using MediaPipe
- Machine learning inference optimization
- Custom algorithm implementation (KNN)
- Database-driven machine learning pipelines

---

## 📧 Contact & Support

**Developed by**: Naufal  
**Email**: [nfatihulx@gmail.com]  
**LinkedIn**: [https://www.linkedin.com/in/naufal-fatihul-6729603a1/]  
**GitHub**: [@nofuruu]

For questions, suggestions, or collaboration inquiries, feel free to reach out!

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - For robust hand detection and landmark extraction
- [OpenCV](https://opencv.org/) - For computer vision operations
- [DearPyGui](https://github.com/hoffstadt/DearPyGui/) - For GPU-accelerated UI rendering
- Community contributions and feedback

---

**Last Updated**: April 2026  
**Version**: 1.0.0
