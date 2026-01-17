# Image Forgery Detection Project 🖼️🔍

## Authors
| Name | GitHub Profile |
| :--- | :--- |
| **Naman Jain** | [@namanviber](https://www.github.com/namanviber) |
| **Dhaval Pathak** | [@Dhaval-pathak](https://github.com/Dhaval-pathak) |
| **Parinita Singh** | [@parinita-singh](https://github.com/parinita-singh) |

**Affiliation**: Computer Science Department, BML Munjal University, Gurgaon, Haryana, India.

---

## Abstract
In the digital age, image manipulation, particularly copy-move forgery, poses a significant challenge, jeopardizing the integrity of digital images. This project focuses on precise forgery detection using the CASIA V2 dataset, employing advanced deep learning techniques, specifically Convolutional Neural Networks (CNNs), and innovative image processing methods. The goal is to accurately identify manipulated sections within images, contributing to digital forensics and ensuring image integrity. The proposed models, Vanilla CNN with Error Level Analysis (ELA) and Dual U-Net model, demonstrate significant advancements in forged region detection.

**Keywords**: Forgery Detection, Deep Learning Techniques, Image Processing Methods, Convolutional Neural Networks (CNNs), Digital Forensics, CASIA V2 Dataset.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Related Work](#related-work)
3. [Methodology](#methodology)
    - [Dataset](#dataset)
    - [Vanilla CNN Model](#vanilla-cnn-model)
    - [Dual U-Net Model](#dual-unet-model)
4. [Results](#results)
    - [Vanilla CNN Model Results](#vanilla-cnn-model-results)
    - [Dual U-Net Model Results](#dual-unet-model-results)
5. [Conclusion](#conclusion)
6. [Project Structure](#-project-structure)
7. [Getting Started](#-getting-started)

---

## Introduction
In the contemporary digital landscape, the rise of powerful image manipulation tools poses a critical challenge to the integrity of digital images. This project addresses the need for reliable methods to verify the authenticity of digital images, especially in the context of widespread misinformation. Leveraging advanced deep learning techniques, including the Vanilla CNN model with Error Level Analysis (ELA) and the Dual U-Net model, the research contributes significantly to the field of image tampering detection.

## Related Work
The proliferation of powerful image manipulation tools has led to an increase in potential misinformation, emphasizing the need for robust methods to authenticate digital images. Previous works have introduced various methodologies, including the BusterNet architecture, GANs and CNNs combination, SIFT technique, and novel techniques for splicing and copy-move forgeries.

## Methodology

### Dataset
For this project, the comprehensive **CASIA V2** dataset was utilized, consisting of 4,795 images with a balanced distribution of authentic and forged instances. The dataset includes diverse manipulation techniques such as copy-move, splicing, and retouching, making it suitable for evaluating and enhancing forgery detection algorithms.

### Vanilla CNN Model
The Vanilla CNN model incorporates **Error Level Analysis (ELA)** during pre-processing, highlighting compression artifacts and potential manipulations. The model architecture comprises convolutional layers, max pooling layers, dropout layers, and dense layers for pattern identification. The performance metrics include accuracy, precision, recall, and F1 Score.

### Dual U-Net Model
The Dual U-Net model, designed for forged region detection, involves pre-processing with resizing and binary segmentation map creation. **SRM filters** are applied to enhance spatial features. The architecture consists of two parallel U-Net networks with skip connections for robust feature localization. Performance is assessed using Intersection over Union (IoU) scores.

## Results

### Vanilla CNN Model Results
The Vanilla CNN model demonstrated superior performance among the evaluated models, achieving:
- **Accuracy**: 92.09%
- **Precision**: 97.16%
- **Recall**: 92.78%
- **F1 Score**: 94.92%

The model effectively detected potential image tampering, as visualized in the confusion matrix, ROC curve, and precision-recall curve.

### Dual U-Net Model Results
The Dual U-Net model showcased significant advancements in forged region detection, with potential applications in image manipulation detection and forgery identification. Performance metrics, including **IoU scores**, highlight the model's efficacy in discerning crucial spatial features.

## Conclusion
This project offers a reliable solution for verifying digital image authenticity and contributes to the broader goal of safeguarding trust and credibility in the digital realm. The proposed models, Vanilla CNN with ELA and Dual U-Net, demonstrate significant advancements in image tampering detection, showcasing their potential applications in diverse domains such as forensics, journalism, and security.

---

## 📂 Project Structure

- `app.py`: Main Streamlit application for image forgery detection.
- `phase1.ipynb` / `phase2.ipynb`: Notebooks containing the training logic for the dual-phase detection system.
- `phase1.h5` / `phase2.h5`: Pre-trained weights for the primary detection and localization models.
- `changeinELA.ipynb` / `data.ipynb`: Experimentation notebooks for ELA processing and data handling.
- `legacy_v1/`: Contains the original implementation with multiple model evaluations.
- `Image Forgery Detection (Group 3).pdf`: Detailed project documentation and report.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- TensorFlow/Keras, OpenCV, Streamlit

### Installation & Run
1. `git clone https://github.com/namanviber/Image-Forgery-Detection-DIP.git`
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

---
🛡️ *Ensuring Digital Integrity with AI.*
