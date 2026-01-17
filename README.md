# Image Forgery Detection 🖼️🔍

Image Forgery Detection is an advanced tool designed to identify manipulated or doctored regions in digital images. This project implements a two-phase approach using deep learning models to detect both the presence of forgery and the specific location of the manipulation.

![Project Overview](Image%20Forgery%20Detection%20(Group%203).pdf) 

## 🌟 Features

- **Double-Phase Verification**: 
    - **Phase 1**: Classifies an image as Real or Forged using Error Level Analysis (ELA).
    - **Phase 2**: Localizes the forged region using a specialized segmentation model.
- **ELA (Error Level Analysis)**: Visualizes pixel inconsistencies indicative of digital manipulation.
- **Region Highlighting**: Provides a visual overlay (heatmap) on the forged areas.
- **Multi-Model Support**: Includes various architectures like VGG16, DenseNet, and MobileNet in the legacy versions.
- **Interactive UI**: User-friendly interface built with Streamlit for easy image uploading and testing.

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- TensorFlow/Keras
- OpenCV
- Streamlit

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/namanviber/Image-Forgery-Detection-DIP.git
   cd Image-Forgery-Detection-DIP
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To launch the current detection tool:
```bash
streamlit run app.py
```

## 📂 Project Structure

- `app.py`: Main Streamlit application for image forgery detection.
- `phase1.ipynb` / `phase2.ipynb`: Notebooks containing the training logic for the dual-phase detection system.
- `phase1.h5` / `phase2.h5`: Pre-trained weights for the primary detection and localization models.
- `changeinELA.ipynb` / `data.ipynb`: Experimentation notebooks for ELA processing and data handling.
- `legacy_v1/`: Contains the original implementation with multiple model evaluations.
    - `streamlit_app.py`: The original app interface.
    - `newVgg16.h5`, `newDenseNet.h5`, etc.: Alternative pre-trained models.
    - `trainModels.ipynb`: Training script for the various architectures.
- `Image Forgery Detection (Group 3).pdf`: Detailed project documentation and report.

## 🧠 Methodology

1. **ELA Processing**: The input image is re-saved at a specific quality to detect compression differences.
2. **Phase 1 (Classification)**: A CNN evaluates the ELA image to determine if digital tampering has occurred.
3. **Phase 2 (Localization)**: If forgery is detected, a segmentation model (using SRM filters and original features) identifies the exact manipulated pixels.

---
🛡️ Ensuring Digital Integrity with AI.
