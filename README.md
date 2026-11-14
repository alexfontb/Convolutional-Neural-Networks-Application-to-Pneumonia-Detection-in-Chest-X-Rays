# Lab 3: Convolutional Neural Networks — Application to Pneumonia Detection in Chest X-Rays

**Authors:**  
- Àlex Font Bartumeus  
- Iris Azlor Solé  

---

## Overview

This repository contains a PyTorch implementation of a Convolutional Neural Network (CNN) designed to classify chest X-ray images into **Normal** (`0`) and **Pneumonia** (`1`) categories. The project covers exploratory data analysis, custom dataset loading from `.npz` archives, model training, and performance evaluation using loss/accuracy curves, confusion matrices, and ROC curves.

> **Note:** The plots displayed inside the notebook reflect a local verification run and may slightly differ from those included in the accompanying lab report (which were generated on Google Colab).

---

## Dataset

The model uses the **PneumoniaMNIST** dataset (`data/pneumonia_MNIST/pneumoniamnist.npz`), consisting of $28 \times 28$ grayscale chest X-ray images with pixel intensities ranging from `0` to `255`.

| Split | Total Images | Normal (`0`) | Pneumonia (`1`) |
| :--- | :---: | :---: | :---: |
| **Train** | 4,708 | 1,214 | 3,494 |
| **Validation** | 524 | 135 | 389 |
| **Test** | 624 | 234 | 390 |
| **Total** | **5,856** | **1,583** | **4,273** |

Images are converted to PIL format, transformed into tensors, and normalized with $\text{mean} = 0.5$ and $\text{std} = 0.5$ via a custom `NPZDataset` class.

---

## Model Architecture & Hyperparameters

### Baseline CNN (`PneumoniaCNN`)
- **Feature Extractor:**
  - `Conv2d(1, 32, kernel_size=3, padding=1)` $\rightarrow$ `ReLU` $\rightarrow$ `MaxPool2d(2)` (outputs $32 \times 14 \times 14$ feature maps)
- **Classifier:**
  - `Flatten` $\rightarrow$ `Linear(6272, 2)`

### Training Configuration
- **Image Dimensions:** $28 \times 28$
- **Batch Size:** `16`
- **Epochs:** `15`
- **Learning Rate:** `0.001` (Adam optimizer)
- **Loss Function:** `CrossEntropyLoss`
- **Random Seed:** `17` (for reproducibility)

---

## Results (Validation Set)

After 15 epochs, the baseline CNN achieved **~97% validation accuracy** (508/524 correct predictions, with 7 false positives and 9 false negatives):

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Normal** | 0.95 | 0.93 | 0.94 | 135 |
| **Pneumonia** | 0.98 | 0.98 | 0.98 | 389 |
| **Overall Accuracy** | | | **0.97** | **524** |

---

## Requirements & Usage

### Dependencies
- `torch` & `torchvision`
- `numpy`
- `matplotlib`
- `Pillow`
- `scikit-learn`

### Running the Notebook
1. Place `pneumoniamnist.npz` inside `data/pneumonia_MNIST/`.
2. Launch Jupyter Notebook or JupyterLab and run all cells sequentially.
