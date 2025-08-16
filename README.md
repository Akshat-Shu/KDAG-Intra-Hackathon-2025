# KDAG_Intras.ipynb – What We Tried

## Introduction

This notebook is basically a record of all the things we tried for the KDAG Intra Hackathon 2025. We experimented with a bunch of different approaches—tabular models, image models, and even hybrid models that use both. Unfortunately, nothing here really worked out: all the models topped out at about 60% accuracy, which is not what we were hoping for.

---

## Data Loading and Exploration

- **Getting the Data:**  
  We used `gdown` to download everything from Google Drive, including the main CSV (`subject_data.csv`) and a big HDF5 file with all the images.

- **First Look:**  
  Loaded the CSV with pandas, checked out the columns, looked for missing values, and tried to get a sense of what features might be useful.

- **Feature Groups:**  
  We organized features into groups like:
    - Diagnosis and target labels
    - Patient info (age, sex, etc.)
    - Lesion geometry
    - Lesion color/texture
    - Other metadata

- **Handling Missing Data:**  
  - Filled missing ages with the median.
  - Filled missing sex with `'unknown'`.

---

## Image Handling

- **Extracting Images:**  
  Used `h5py` and PIL to pull images out of the HDF5 and save them as JPEGs. This made loading them later much faster.

- **Visual Checks:**  
  Plotted random images and some specific cases (like all malignant ones) just to see what the data looked like and if there were any obvious issues.

---

## Tabular Data Preprocessing

- **Feature Engineering:**  
  Picked out which features to use, scaled the numerical ones, and one-hot encoded the categorical ones.

- **Label Encoding:**  
  Used pandas' `factorize` to turn diagnosis strings into numbers for the ML models.

---

## Modeling Attempts

### 1. Tabular-Only Models

- **What We Tried:**  
  - Hierarchical classifiers with the `hiclass` library.
  - LightGBM and XGBoost.

- **How It Went:**  
  - Accuracy hovered around 60%. The tabular data alone just didn’t have enough signal.

---

### 2. Image-Only Models

- **What We Tried:**  
  - Built a basic CNN, ResNet18, and EfficientNetB0 in PyTorch.
  - Used standard augmentations and normalization.

- **How It Went:**  
  - Even with class balancing and all, these models also capped out at about 60% accuracy.

---

### 3. Hybrid (Tabular + Image) Models

- **What We Tried:**  
  - Made a custom PyTorch Dataset that returns both image and tabular features.
  - Built a hybrid model: EfficientNet or ViT for images, MLP for tabular, then concatenate and classify.

- **How It Went:**  
  - Still stuck at ~60% accuracy. Combining both types of data didn’t really help.

---

## Model Evaluation

- **Metrics:**  
  We tracked accuracy, precision, recall, specificity, F1, and AUC.

- **Findings:**  
  No matter what we tried, nothing broke past the 60% barrier. Either the features just aren’t strong enough, or we need to rethink our approach.

---

## Summary

- **We tried:**  
  - Tabular-only models (LightGBM, XGBoost, hierarchical)
  - Image-only deep learning (CNNs, EfficientNet, ViT)
  - Hybrid models (tabular + image)

- **Result:**  
  Everything in this notebook gave unsatisfactory results—accuracy stuck at ~60%. We’ll need to try some more creative feature engineering or maybe look for better data.