
# 😷 Face Mask Detection System – Real-Time Deep Learning with Streamlit & OpenCV

This project is a deep learning-based **face mask detection system** that classifies whether individuals in an image or video stream are wearing a mask. Powered by **MobileNetV2**, it supports both **real-time webcam detection** and a **Streamlit web interface** for static image uploads. The project is built end-to-end with training, evaluation, visualization, and deployment components.

---

## 🚀 Key Features

- 🤖 **Real-time Face Mask Detection** via webcam (OpenCV + DNN)
- 🖼️ **Streamlit Web App** to upload images and annotate faces with mask status
- 🧠 **Transfer Learning** with `MobileNetV2` pretrained on ImageNet
- 📉 **Two-Phase Training Strategy**: Head training + fine-tuning top layers
- 📊 Includes **ROC, PR Curve, Confusion Matrix**, and **Classification Report**
- 💾 Saves models in `.h5` format for reuse and deployment

---

## 🧠 Model Architecture
- Base: `MobileNetV2` (frozen in phase-1, partially unfrozen in phase-2)
- Added Layers: `GlobalAveragePooling2D` → `Dropout(0.2)` → `Dense(1, sigmoid)`
- Loss: `Binary Crossentropy`
- Optimizers: `Adam (1e-4 → 1e-5)`

---

## 🧪 Evaluation Highlights
- Accuracy scale interpretation (Exceptional, Excellent, Good, etc.)
- Full classification report and confusion matrix for validation data
- ROC-AUC and Precision–Recall curve generation included

---

## 🧾 Project Structure
```
├── train_mask_detector.py      # Main training pipeline
├── train_and_evaluate.py       # Extended evaluation with metrics
├── load_data.py                # Image generator with validation split
├── app.py                      # Streamlit web interface
├── detect_mask_video.py        # Webcam real-time detection
├── mask_detector_final.h5      # Final trained model (saved during training)
```

---

## 🖥️ How to Use

### 📦 Clone and Install
```bash
git clone https://github.com/Akkkkkansha/face-mask-detector.git
cd face-mask-detector
pip install -r requirements.txt
```

### ▶️ Run the Streamlit App
```bash
streamlit run app.py
```

### 🎥 Run the Webcam-Based Detector
```bash
python detect_mask_video.py
```

---

## 👩‍💻 Author

**Akansha**  
Aspiring AI Developer | Passionate about building real-time AI systems for real-world impact  
🔗 [GitHub](https://github.com/Akkkkkkansha) | 📍 Delhi, India

---

## 🧠 Keywords

`Face Mask Detection` `Deep Learning` `MobileNetV2` `OpenCV` `Streamlit` `Real-Time AI` `Binary Classification` `Transfer Learning` `Computer Vision`
