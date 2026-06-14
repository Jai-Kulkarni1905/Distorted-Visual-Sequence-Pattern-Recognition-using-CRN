# Distorted CAPTCHA Recognition using CRNN

A deep learning-based Optical Character Recognition (OCR) system developed for recognizing highly distorted CAPTCHA-style text sequences from grayscale images. The project combines Convolutional Neural Networks (CNNs) for visual feature extraction with Bidirectional LSTMs (BiLSTMs) for sequence modeling and Connectionist Temporal Classification (CTC) Loss for alignment-free text prediction. The objective is to accurately reconstruct hidden character sequences despite noise, blur, overlap, deformation, and occlusion. :contentReference[oaicite:0]{index=0}

---

## Project Overview

Traditional OCR systems struggle when characters are distorted, overlapping, partially hidden, or affected by visual artifacts. This project addresses these challenges by implementing a Convolutional Recurrent Neural Network (CRNN) capable of learning both spatial and sequential patterns directly from CAPTCHA images.

The complete workflow includes:

- Data validation and preprocessing
- Character vocabulary generation
- Exploratory data analysis (EDA)
- Custom PyTorch datasets and dataloaders
- CRNN architecture implementation
- Training using CTC Loss
- Validation and evaluation
- Test set inference and submission generation

---

## Dataset

The dataset consists of grayscale CAPTCHA images and corresponding text labels.
The link to the dataset : https://drive.google.com/drive/folders/1lRUA-1uCCXfks8kpypFV-4f0UepWoLkU?usp=sharing
### Training Set
- CAPTCHA images
- Ground truth text labels

### Test Set
- CAPTCHA images only

Each image contains a six-character alphanumeric sequence affected by various distortions including:

- Background noise
- Blur and artifacts
- Character overlap
- Shape deformation
- Occlusion
- Irregular spacing

---

## Model Architecture

### CRNN Pipeline

Input Image
→ CNN Feature Extractor
→ Sequence Conversion
→ Bidirectional LSTM Layers
→ Linear Projection
→ CTC Decoder
→ Predicted Text Sequence

### Key Components

- CNN layers for visual feature extraction
- BiLSTM layers for sequence learning
- CTC Loss for alignment-free training
- Greedy decoding during inference

---

## Exploratory Analysis

The notebook includes:

- Sample image visualization
- Image dimension analysis
- Label validation and cleaning
- Character frequency analysis
- Vocabulary generation
- Batch verification and data inspection

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Pandas
- OpenCV
- Matplotlib
- Scikit-learn

---

## Project Structure

```text
├── notebook_jai_23119015.ipynb
├── train_images/
├── test_images/
├── train-labels.csv
├── submission.csv
└── README.md
