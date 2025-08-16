# KDAG_Intras.ipynb – What We Tried

## Introduction

This notebook is basically a record of all the things we tried for the KDAG Intra Hackathon 2025. We experimented with a bunch of different approaches—tabular models, image models, and even hybrid models that use both. Unfortunately, nothing here really worked out: all the models topped out at about 60% accuracy, which is not what we were hoping for.



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



## Image Handling

- **Extracting Images:**  
  Used `h5py` and PIL to pull images out of the HDF5 and save them as JPEGs. This made loading them later much faster.

- **Visual Checks:**  
  Plotted random images and some specific cases (like all malignant ones) just to see what the data looked like and if there were any obvious issues.



## Tabular Data Preprocessing

- **Feature Engineering:**  
  Picked out which features to use, scaled the numerical ones, and one-hot encoded the categorical ones.

- **Label Encoding:**  
  Used pandas' `factorize` to turn diagnosis strings into numbers for the ML models.



## Modeling Attempts

### 1. Tabular-Only Models

- **What We Tried:**  
  - Hierarchical classifiers with the `hiclass` library.
  - LightGBM and XGBoost.

- **How It Went:**  
  - Accuracy hovered around 60%. The tabular data alone just didn’t have enough signal.



### 2. Image-Only Models

- **What We Tried:**  
  - Built a basic CNN, ResNet18, and EfficientNetB0 in PyTorch.
  - Used standard augmentations and normalization.

- **How It Went:**  
  - Even with class balancing and all, these models also capped out at about 60% accuracy.



### 3. Hybrid (Tabular + Image) Models

- **What We Tried:**  
  - Made a custom PyTorch Dataset that returns both image and tabular features.
  - Built a hybrid model: ViT for images, MLP for tabular, then concatenate and classify.

- **How It Went:**  
  - Still stuck at ~60% accuracy. Combining both types of data didn’t really help.



## Model Evaluation

- **Metrics:**  
  We tracked accuracy, precision, recall, specificity, F1, and AUC.

- **Findings:**  
  No matter what we tried, nothing broke past the 60% barrier. Either the features just aren’t strong enough, or we need to rethink our approach.



## Summary

- **We tried:**  
  - Tabular-only models (LightGBM, XGBoost, hierarchical)
  - Image-only deep learning (CNNs, EfficientNet, ViT)
  - Hybrid models (tabular + image)

- **Result:**  
  Everything in this notebook gave unsatisfactory results—accuracy stuck at ~60%. We’ll need to try some more creative feature engineering or maybe look for better data.


# Solution_Model.ipynb – Final Solution and External Model Experiments

## Introduction

This notebook is where we put together our final solution pipeline for the KDAG Intra Hackathon 2025. We also tried out some external models for comparison. After all the trial and error in the previous notebook, here we focused on refining our preprocessing, using more advanced architectures, and seeing how our results stacked up against models made by others.



## Data Preparation

- **Data Download:**  
  We used `gdown` and `wget` to grab both the dataset and some external models (like a Keras `.h5` file) from Google Drive and GitHub.

- **Feature Selection:**  
  Picked out a set of `meta_features` based on what seemed most promising from earlier experiments and domain knowledge.

- **Image Extraction:**  
  Extracted images from the HDF5 file and saved them as JPEGs for faster access during training and inference.



## Trying Out External Models

- **Loading External Models:**  
  Downloaded a pre-trained Keras model (`mymodel-2.h5`) and loaded it with TensorFlow/Keras.

- **Compatibility Fixes:**  
  Had to make sure the right versions of TensorFlow and Keras were installed so the model would actually load.

- **Testing:**  
  - Resized images to the input size expected by the model.
  - Ran predictions on both malignant and benign samples.
  - Calculated metrics like accuracy, sensitivity, specificity, PPV, NPV, and F1-score.
  - Plotted confusion matrices and ROC curves to visualize performance.



## Our Custom PyTorch Solution

### 1. Custom Dataset and Augmentations

- **CancerDataset:**  
  Built a PyTorch Dataset that returns both image and meta features, handles missing values, and encodes categorical variables.

- **Augmentations:**  
  Used `albumentations` for strong image augmentations during training, and standard normalization and resizing for validation.



### 2. Model Architecture

- **EfficientNet Backbone:**  
  Used the `geffnet` library for EfficientNet variants. Combined image features with meta features using a small MLP, and added multiple dropout layers for regularization.

- **Custom Swish Activation:**  
  Implemented a custom Swish activation function for a slight performance boost.



### 3. Training Strategy

- **Class Imbalance Handling:**  
  Used `WeightedRandomSampler` and calculated class weights to make sure the model didn't just learn to predict the majority class.

- **Learning Rate Scheduling:**  
  Combined cosine annealing with gradual warmup for more stable training.

- **Cross-Validation:**  
  Used stratified K-fold cross-validation to get a more reliable estimate of model performance.



### 4. Evaluation

- **Metrics:**  
  Calculated accuracy, sensitivity, specificity, PPV, NPV, F1-score, and AUC. Also visualized confusion matrices and ROC curves.

- **Model Saving:**  
  Saved the best model weights based on validation AUC so we could reload them later.



## Key Takeaways

- **External Models:**  
  Gave us a useful benchmark and showed that this dataset is genuinely tough.

- **Our Solution:**  
  Combining EfficientNet with meta features, using strong augmentations, and balancing the classes helped us do better than our earlier attempts.

- **Advanced Tricks:**  
  Using custom learning rate schedules, hybrid architectures, and careful data handling made a noticeable difference.



## Conclusion

- We started with basic models and gradually made things more complex, learning from each step.
- Careful preprocessing, augmentation, and class balancing were just as important as the model architecture itself.
- Comparing our results with external models helped us set realistic expectations and figure out where to focus our efforts next.



**Note:**  
All the models in `KDAG_Intras.ipynb` gave unsatisfactory results (about 60% accuracy). This notebook (`Solution_Model.ipynb`) documents our improved pipeline and experiments with external models, which gave us better results.



# Solution_Model_Modified.ipynb – Adds another branch to the existing solution model

## Introduction

Deeper analysis of the images shows that, the malignant cases are further subdivided into 3 classes, namely basel cell carcinoma, melanoma, squamous cell carcinoma, now we can use this classification to further imporove the model by extracting features from each subclass, doing this leads to tighter boundries around the malignant cases, hence improving the accuracy of the benign cases.

## Model Architecture

In the above model, we use a shared EfficientNet backbone to extract image features. From these features, we add another branch to predict the 3 malignant subtypes, while the main branch combines the image features, the subtype-projected features, and the tabular metadata. Finally, this concatenated representation is used to predict whether the case is benign or malignant.

## Kaggle pretrained model links
- [Kaggle Notebook](https://www.kaggle.com/code/joshuarajtadi/solution-model-modified)
- [GDrive link for model weights](https://drive.google.com/file/d/1-GaHMgmFkQvbVkt1iKmvzUhABeStglUw/view?usp=sharing)
## Result

This model reduces the number of false positives, which means fewer patients are unnecessarily alarmed, but the trade-off is that it increases the number of false negatives, which is quite fatal for this type of cancer predictor.

## Conclusion

We could use both the solution model, as the first one predicts the malignant cases more accurately, while the second one predicts the benign cases more accurately.
